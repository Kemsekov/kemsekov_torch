import torch
from torch import nn

# initializes module last layer weight
def init_last_layer_weight(module : nn.Module,const):
    reverse_subm = list(module.children())[::-1] + [module]
    for m in reverse_subm:
        if hasattr(m,'weight') and isinstance(m.weight,nn.parameter.Parameter):
            nn.init.constant_(m.weight,const)
            break
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1,dilation=2,bias=None,padding_mode='zeros',conv2d_impl = nn.Conv2d):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super().__init__()
        modulation=True
        adaptive_d=True
        
        inc=in_channels
        outc=out_channels
        self.outc=outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation=dilation
        self.adaptive_d= adaptive_d
        
        #useless thing
        self._sum = 0.0

        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = conv2d_impl(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias,padding_mode=padding_mode)

        self.p_conv = conv2d_impl(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride,padding_mode=padding_mode)

        init_last_layer_weight(self.p_conv, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = conv2d_impl(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride,padding_mode=padding_mode)
            init_last_layer_weight(self.m_conv, 0.5)
            self.m_conv.register_backward_hook(self._set_lr)
        else:
            self.m_conv=torch.nn.Identity()
            
        if self.adaptive_d:
            self.ad_conv = conv2d_impl(inc,kernel_size,kernel_size=3,padding=1,stride=stride,padding_mode=padding_mode)
            init_last_layer_weight(self.ad_conv,1)
            self.ad_conv.register_backward_hook(self._set_lr)
        else:
            self.ad_conv=torch.nn.Identity()

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # print(len(grad_input),len(grad_output))
        # print(grad_input,grad_output)
        # print(["NONE" if v is None else v.shape for v in grad_input],["NONE" if v is None else v.shape for v in grad_output])
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
        
    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        
        # (b, 2N, h, w)
        ad_base = self.ad_conv(x)
        ad_base = 1-torch.sigmoid(ad_base)
        ad = ad_base.repeat(1,2*self.kernel_size,1,1)*self.dilation
        
        ad_m = (ad_base - 0.5)*2 
        ad_m = ad_m.repeat(1,self.kernel_size,1,1)*self.dilation
        p = self._get_p(offset, dtype,ad)

        m = torch.sigmoid(self.m_conv(x))

        if self.padding:
            x = self.zero_padding(x)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.data.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        slice_p = p[..., :N]
        slice_p_b = p[..., N:]
        
        # bilinear kernel (b, h, w, N)
        qlt_slice1,qlt_slice2   = q_lt[..., :N].type_as(p) - slice_p, (q_lt[..., N:].type_as(p) - slice_p_b)
        q_rb_slice1,q_rb_slice2 = q_rb[..., :N].type_as(p) - slice_p, (q_rb[..., N:].type_as(p) - slice_p_b)
        q_lb_slice1,q_lb_slice2 = q_lb[..., :N].type_as(p) - slice_p, (q_lb[..., N:].type_as(p) - slice_p_b)
        q_rt_slice1,q_rt_slice2 = q_rt[..., :N].type_as(p) - slice_p, (q_rt[..., N:].type_as(p) - slice_p_b)

        # print(q_rt_slice1.shape==q_rt_slice2.shape)

        g_lt = (1 + qlt_slice1)  * (1 + qlt_slice2)
        g_rb = (1 - q_rb_slice1) * (1 - q_rb_slice2)
        g_lb = (1 + q_lb_slice1) * (1 - q_lb_slice2)
        g_rt = (1 - q_rt_slice1) * (1 + q_rt_slice2)

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        
        m = m*ad_m
        
        m = m.contiguous().permute(0, 2, 3, 1)
        m = m.unsqueeze(dim=1)
        m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
        x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        
        # --------------------------------------------------
        # i have no idea why torch breaks when i remove this useless agg value
        # useless thing
        agg = torch.mean(qlt_slice1+q_rb_slice1+q_lb_slice1+q_rt_slice1)*1e-20
        self._sum = 0.0
        self._sum+=float(agg.item())
        # --------------------------------------------------
        
        return out

    def _get_p_n(self, N : int, dtype):
        #st=torch.cat([s_dilation,torch.zeros(1).cuda(),torch.flip(-s_dilation,dims=[0])],dim=0).type(dtype)
        #st=st.repeat(self.kernel_size,1)*self.dilation
        #st= torch.clamp_min(st,0)


        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)

        p_n_x = p_n_x.type(dtype)
        p_n_y = p_n_y.type(dtype)

        #st_x = st*p_n_x
        #st_y = st*p_n_y

        #p_n_x = p_n_x + st_x
        #p_n_y = p_n_y +st_y

        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h : int, w  : int, N  : int, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype, ad_offset):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset + ad_offset*p_n
        return p
    def _get_p_without_ad(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N : int):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks : int):
        b, c, h, w, N = x_offset.size()
        N = int(N)
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset