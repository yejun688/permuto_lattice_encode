import torch
import torch.nn.functional as F
import numpy as np

# 该函数通常用于渲染和光线追踪中。在渲染中，用于计算表面点的局部坐标系。在光线追踪中，用于计算射线与表面相交点的坐标系。
def create_frame(n: torch.Tensor, eps:float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    :param: n (bn, 3, ...)
    """
    z = F.normalize(n, dim=1, eps=eps)
    sgn = torch.where(z[:,2,...] >= 0, 1.0, -1.0)
    a = -1.0 / (sgn + z[:,2,...])
    b = z[:,0,...] * z[:,1,...] * a
    x = torch.stack([1.0 + sgn * z[:,0,...] * z[:,0,...] * a, sgn * b, -sgn * z[:,0,...]], dim=1)
    y = torch.stack([b, sgn + z[:,1,...] * z[:,1,...] * a, -z[:,1,...]], dim=1)
    return x, y, z


# 该函数通常用于渲染和光线追踪中。在渲染中，用于计算表面点的渲染参数。在光线追踪中，用于计算射线与表面相交点的渲染参数。
def get_rendering_parameters(albedo_raw, rough_raw, use_metallic):
    if use_metallic:
        assert albedo_raw.size(-1) == 3 and rough_raw.size(-1) == 2
        metal = rough_raw[:,1:]
        rough = rough_raw[:,:1].clamp_min(0.01)
        Ks = baseColorToSpecularF0(albedo_raw, metal)
        Kd = albedo_raw * (1 - metal)
    else:
        assert albedo_raw.size(-1) == 6 and rough_raw.size(-1) == 1
        Kd = albedo_raw[:,:3]
        Ks = albedo_raw[:,3:].clamp_min(0.04)
        rough = rough_raw.clamp_min(0.01)
    return Kd, Ks, rough


def to_global(d, x, y, z):
    """
    d, x, y, z: (*, 3)
    """
    return d[...,0:1] * x + d[...,1:2] * y + d[...,2:3] * z

def sqrt_(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    clamping 0 values of sqrt input to avoid NAN gradients
    """
    return torch.sqrt(torch.clamp(x, min=eps))

# reflect() 函数用于计算向量 v 在向量 h 上的反射。
def reflect(v: torch.Tensor, h: torch.Tensor):
    dot = torch.sum(v*h, dim=2, keepdim=True)
    return 2 * dot * h - v

# square_to_cosine_hemisphere() 函数用于从方形采样空间转换为球形采样空间
def square_to_cosine_hemisphere(sample: torch.Tensor):
    #
    u, v = sample[:,:,0,...], sample[:,:,1,...]
    phi = u * 2 * np.pi
    r = sqrt_(v)
    cos_theta = sqrt_(torch.clamp(1 - v, 0))
    return torch.stack([torch.cos(phi) * r, torch.sin(phi) * r, cos_theta], dim=2)


def get_cos_theta(v: torch.Tensor):
    return v[:,:,2,...]


# get_phi() 函数用于从球形采样空间中的采样点计算方位角。
def get_phi(v: torch.Tensor):
    cos_theta = torch.clamp(v[:,:,2,...], min=0, max=1)
    sin_theta = torch.clamp(sqrt_(1 - cos_theta*cos_theta), min=1e-8)
    cos_phi = torch.clamp(v[:,:,0,...] / sin_theta, -1, 1)
    sin_phi = v[:,:,1,...] / sin_theta
    phi = torch.acos(cos_phi) # (0, pi)
    return torch.where(sin_phi > 0, phi, 2*np.pi - phi)


# 计算迪士尼镜面反射模型中反射光线的方向,将反射光线方向归一化
# 该函数可以用于任何场景，需要从迪士尼镜面反射模型中采样光线方向的地方。
# 该函数可以用于光线追踪和渲染中。在光线追踪中，用于计算反射光线的方向。在渲染中，用于计算材质的光泽效果。
def sample_disney_specular(sample: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor):
    """
    :param: sample (bn, spp, 3, h, w)
    :param: roughness (bn, 1, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), phi (bn, spp, h, w), cos theta (bn, spp, h, w)
    """
    # a = torch.clamp(roughness, 0.001)
    a = roughness
    u, v = sample[:,:,0,...], sample[:,:,1,...]
    phi = u * 2 * np.pi
    cos_theta = sqrt_((1 - v) / (1 + (a*a - 1) * v))
    sin_theta = sqrt_(1 - cos_theta*cos_theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    # half 是入射光线方向和反射光线方向的和,
    half = torch.stack([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta], dim=2)
    # wo 是反射光线方向,将反射光线方向归一化,
    # reflect() 函数用于计算反射光线方向
    wo = F.normalize(reflect(wi.expand_as(half), half), dim=2, eps=1e-8)
    return wo
    #, phi.squeeze(2), cos_theta.squeeze(2)



def GTR2(ndh, a):
    a2 = a*a
    t = 1.0 + (a2 - 1.0) * ndh * ndh
    return a2 / (np.pi * t * t)

# SchlickFresnel() 函数用于计算光线与表面相交时的菲涅尔反射系数
def SchlickFresnel(u):
    m = torch.clamp(1.0 - u, 0, 1)
    return m**5

# 计算 GGX 粗糙度模型中的 G 函数
def smithG_GGX(ndv, a):
    a = a*a
    b = ndv*ndv
    return 1.0 / (ndv + sqrt_(a + b - a * b))


# pdf_disney() 函数用于计算迪士尼材质的 PDF（概率密度函数）
def pdf_disney(roughness: torch.Tensor, metallic: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    """
    # specularAlpha = torch.clamp(roughness, 0.001)
    specularAlpha = roughness
    diffuseRatio = 0.5 * (1 - metallic)
    specularRatio = 1 - diffuseRatio
    half = F.normalize(wi + wo, dim=2, eps=1e-8)
    cosTheta = torch.abs(half[:,:,2,...])
    # 镜面反射 PDF 的计算方法如下
    pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta
    pdfSpec = pdfGTR2 / torch.clamp(4.0 * torch.abs(torch.sum(wo*half, dim=2)), min=1e-8)
    # 漫反射 PDF 的计算方法如下：
    #  是反射光线方向与法向量的夹角的绝对值
    pdfDiff = torch.abs(wo[:,:,2,...]) / np.pi
    # 最终 PDF 的计算方法如下:
    pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec
    pdf = torch.where(wi[:,:,2,...] < 0.0001, torch.ones_like(pdf) * 0.0001, pdf)
    pdf = torch.where(wo[:,:,2,...] < 0.0001, torch.ones_like(pdf) * 0.0001, pdf)
    return pdf


#wi：入射光线方向;   wo：反射光线方向
def eval_disney(albedo: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: albedo/roughness/metallic (bn, c, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    """
    h = wi + wo;
    h = F.normalize(h, dim=2, eps=1e-8)
    # CSpec0 是一个用于计算镜面反射系数 Fs 的参数
    CSpec0 = torch.lerp(torch.ones_like(albedo)*0.04, albedo, metallic).unsqueeze(1)

    ldh = torch.clamp(torch.sum( (wo * h), dim = 2), 0, 1).unsqueeze(2)
    ndv = wi[:,:,2:3,...]
    ndl = wo[:,:,2:3,...]
    ndh = h[:,:,2:3,...]

    FL, FV = SchlickFresnel(ndl), SchlickFresnel(ndv)
    roughness = roughness.unsqueeze(1)
    Fd90 = 0.5 + 2.0 * ldh * ldh * roughness
    Fd = torch.lerp(torch.ones_like(Fd90), Fd90, FL) * torch.lerp(torch.ones_like(Fd90), Fd90, FV)

    Ds = GTR2(ndh, roughness)
    FH = SchlickFresnel(ldh)
    # torch.lerp() 函数用于线性插值
    Fs = torch.lerp(CSpec0, torch.ones_like(CSpec0), FH)
    roughg = (roughness * 0.5 + 0.5) ** 2
    Gs1, Gs2 = smithG_GGX(ndl, roughg), smithG_GGX(ndv, roughg)
    Gs = Gs1 * Gs2

    eval_diff = Fd * albedo.unsqueeze(1) * (1.0 - metallic.unsqueeze(1)) / np.pi
    eval_spec = Gs * Fs * Ds
    mask = torch.where(ndl < 0, torch.zeros_like(ndl), torch.ones(ndl))
    return eval_diff, eval_spec, mask


# 用于计算 Schlick Fresnel 系数。Schlick Fresnel 系数用于计算光线在材质表面反射的比例
def F_Schlick(SpecularColor, VoH):
	Fc = (1 - VoH)**5
	return torch.clamp(50.0 * SpecularColor[:,:,1:2,...], min=0, max=1) * Fc + (1 - Fc) * SpecularColor

# 函数用于计算镜面事件的概率
def GetSpecularEventProbability(SpecularColor, NoV) -> torch.Tensor:
	f = F_Schlick(SpecularColor, NoV);
	return (f[:,:,0,...] + f[:,:,1,...] + f[:,:,2,...]) / 3

# baseColorToSpecularF0() 函数用于将材质的基色转换为镜面反射的 F0 值。F0 值是指光线在材质表面发生镜面反射时，入射角为 0 时反射光线的颜色
#
def baseColorToSpecularF0(baseColor, metalness):
    return torch.lerp(torch.empty_like(baseColor).fill_(0.04), baseColor, metalness)

# luminance() 函数用于计算颜色的明亮度。明亮度是指颜色的亮度或鲜艳度
def luminance(color):
    if color.size(1) == 1:
        return color
    # return color.mean(dim=1, keepdim=True)
    return color[:,0:1,...] * 0.212671 + color[:,1:2,...] * 0.715160 + color[:,2:3,...] * 0.072169

# 计算镜面反射采样的概率
def probabilityToSampleSpecular(difColor, specColor) -> torch.Tensor:
    lumDiffuse = torch.clamp(luminance(difColor), min=0.01)
    lumSpecular = torch.clamp(luminance(specColor), min=0.01)
    return lumSpecular / (lumDiffuse + lumSpecular)

# 函数用于计算在 90 度入射角时，被阴影遮挡的镜面反射颜色。
def shadowedF90(F0):
    t = 1 / 0.04
    return torch.clamp(t * luminance(F0), max=1)

#  evalFresnel() 函数用于计算 Fresnel 系数。Fresnel 系数用于计算光线在材质表面发生镜面反射时的反射率
def evalFresnel(f0, f90, NdotS):
    # print(f0.shape, f90.shape, NdotS.shape)
    return f0 + (f90 - f0) * (1 - NdotS)**5

# Smith_G1_GGX() 函数用于计算 Smith GGX 分布函数。Smith GGX 分布函数用于计算光线在漫反射表面发生漫反射时的概率
def Smith_G1_GGX(alphaSquared, NdotSSquared):
    return 2 / (sqrt_(((alphaSquared * (1 - NdotSSquared)) + NdotSSquared) / NdotSSquared) + 1)

# 函数用于计算 Smith GGX 分布函数的二次项。Smith GGX 分布函数的二次项用于计算光线在漫反射表面发生漫反射时的散射程度。
def Smith_G2_GGX(alphaSquared, NdotL, NdotV):
	a = NdotV * sqrt_(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL))
	b = NdotL * sqrt_(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV))
	return 0.5 / (a + b)

# GGX_D() 函数用于计算 GGX 分布函数。GGX 分布函数用于计算光线在粗糙表面发生漫反射时的概率。
def GGX_D(alphaSquared, NdotH):
    b = ((alphaSquared - 1) * NdotH * NdotH + 1)
    return alphaSquared / (np.pi * b * b)

# 该函数返回一个形状为 (*, *, h, w) 的张量，表示 GGX 分布函数的 PDF 的值
def pdf_ggx(Kd: torch.Tensor, Ks: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, ps_min=0.0):
    """
    :param: color (bn, 3, h, w)
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :param: wo (*, *, 3, h, w), supposed to be normalized
    :return: pdf (*, *, h, w)
    """
    alpha = roughness * roughness
    alphaSquared = alpha * alpha
    NdotV = wi[:,:,2,...]
    h = F.normalize(wi + wo, dim=2, eps=1e-8)
    NdotH = h[:,:,2,...]
    # print(alphaSquared.min(), NdotH.min(), NdotV.min())
    ggxd = GGX_D(torch.clamp(alphaSquared, min=0.00001), NdotH)
    smith = Smith_G1_GGX(alphaSquared, NdotV * NdotV)
    # pdf_spec = GGX_D(torch.clamp(alphaSquared, min=0.00001), NdotH) * Smith_G1_GGX(alphaSquared, NdotV * NdotV) / (4 * NdotV)
    pdf_spec = ggxd * smith / (4 * NdotV)
    # print(torch.any(torch.isnan(ggxd)), torch.any(torch.isnan(smith)), torch.any(torch.isnan(NdotV)))
    # print(NdotV.min(), ggxd.min(), smith.min())
    with torch.no_grad():
        pS = probabilityToSampleSpecular(Kd, Ks).clamp_min(ps_min)
    pdf_diff = wo[:,:,2,...] / np.pi
    # print("#########################################")
    # print("#########################################")
    # print("#########################################")
    # print(torch.any(torch.isnan(kS)), torch.any(torch.isnan(pdf_spec)), torch.any(torch.isnan(pdf_diff)))
    # print("#########################################")
    # print("#########################################")
    # print("#########################################")
    pdf = pS * pdf_spec + (1 - pS) * pdf_diff
    pdf = torch.where(wi[:,:,2,...] <= 0.0001, torch.ones_like(pdf) * 0.0001, pdf)
    pdf = torch.where(wo[:,:,2,...] <= 0.0001, torch.ones_like(pdf) * 0.0001, pdf)
    return pdf

# eval_ggx() 函数用于计算 GGX 分布函数的微分反射率（BRDF）。BRDF 用于计算光线在材质表面发生反射时的反射率
def eval_ggx(Kd: torch.Tensor, Ks: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor):
    """
    :param: color (bn, c, h, w)
    :param: roughness/metallic (bn, 1, h, w)
    :param: wi (*, *, c, h, w), supposed to be normalized
    :param: wo (*, *, c, h, w), supposed to be normalized
    :return: fr(wi, wo) (*, *, c, h, w)
    """
    # 计算光线入射方向、视线方向和表面法向量的夹角
    NDotL = wo[:,:,2:3,...]
    NDotV = wi[:,:,2:3,...]
    # 计算光线入射方向和视线方向的半程向量
    H = F.normalize(wi + wo, dim=2, eps=1e-8)
    # 计算光线入射方向、半程向量和表面法向量的夹角
    NDotH = H[:,:,2:3,...]
    # 计算光线视线方向和半程向量的点积
    LDotH = torch.sum(wo*H, dim=2, keepdim=True)
    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    # 计算 GGX 分布函数的 PDF
    D = GGX_D(torch.clamp(alpha2, min=0.00001), NDotH)
    G2 = Smith_G2_GGX(alpha2, NDotL, NDotV)
    f = evalFresnel(Ks.unsqueeze(1), shadowedF90(Ks).unsqueeze(1), LDotH)
    # spec = torch.where(NDotL <= 0, torch.zeros_like(NDotL), f * G2 * D)
    # mask = torch.where(NDotL <= 0, torch.zeros_like(NDotL), torch.ones_like(NDotL))
    spec = torch.where(NDotL < 0.0001, torch.ones_like(NDotL) * 0.0001, f * G2 * D)
    # mask = torch.where(NDotL <= 0, torch.zeros_like(NDotL), torch.ones_like(NDotL))
    mask = (NDotL >= 0.0001).squeeze(-1)
    # 该函数返回三个张量，分别表示材质的漫反射 BRDF（形状为 (*, *, c, h, w)）、镜面反射 BRDF（形状为 (*, *, c, h, w)）和遮罩（形状为 (*, *, h, w)）。
    return Kd.unsqueeze(1) / np.pi, spec, mask


# sample_weight_ggx() 函数用于计算 GGX 分布函数的采样权重。采样权重用于计算光线在材质表面发生反射时的概率
# sample_weight_ggx() 函数主要用于计算 GGX 分布函数的采样权重。该函数被广泛应用于计算机图形学中，例如渲染光照和反射效果。
def sample_weight_ggx(alphaSquared, NdotL, NdotV):
    # alphaSquared：材质的粗糙度参数的平方，形状为 (*, 1, 1, h, w)
    G1V = Smith_G1_GGX(alphaSquared, NdotV*NdotV)
    G1L = Smith_G1_GGX(alphaSquared, NdotL*NdotL)
    return G1L / (G1V + G1L - G1V * G1L)

# sample_ggx() 函数用于采样 GGX 镜面反射 BRDF。
def sample_ggx(sample: torch.Tensor, Kd: torch.Tensor, Ks: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor):
    """
    :param: sample (bn, spp, 3, h, w)
    :param: roughness (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), weight (bn, spp, 3, h, w)
    """
    with torch.no_grad():
        pS = probabilityToSampleSpecular(Kd, Ks)
    sample_diffuse = sample[:,:,2,...] >= pS
    # 这段代码用于将采样点的位置从正方形半球体转换到余弦半球体。
    wo_diff = square_to_cosine_hemisphere(sample[:,:,1:,...])
    # 这段代码用于计算漫反射的采样权重
    weight_diff = Kd / (1 - pS)
    # 这段代码用于将漫反射的采样权重扩展为一个三维张量
    weight_diff = weight_diff.unsqueeze(1)
    # 将粗糙度扩展为一个三维张量
    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    # alpha = roughness
    # 这段代码用于计算虚拟半向量 Vh。虚拟半向量是入射光线方向和出射光线方向的加权平均值。它用于计算镜面反射的 BRDF
    Vh = F.normalize(torch.cat([alpha * wi[:,:,0:1,...], alpha * wi[:,:,1:2,...], wi[:,:,2:3,...]], dim=2), dim=2, eps=1e-8)
    # 这段代码用于计算虚拟半向量的长度平方 lensq
    lensq = Vh[:,:,0:1,...]**2 + Vh[:,:,1:2,...]**2
    # 这段代码用于创建两个零向量和一个一向量。
    zero_ = torch.zeros_like(Vh[:,:,0,...])
    one_ = torch.ones_like(Vh[:,:,0,...])
    # 这段代码用于计算一个张量 T1，该张量表示反射方向
    T1 = torch.where(
        lensq > 0, 
        torch.stack([-Vh[:,:,1,...], Vh[:,:,0,...], zero_], dim=2) / sqrt_(lensq),
        torch.stack([one_, zero_, zero_], dim=2)
    )
    # 代码用于计算一个张量 T2，该张量表示反射方向的正交向量
    T2 = torch.cross(Vh, T1, dim=2)
    # 这段代码用于计算采样点的极坐标
    r = sqrt_(sample[:,:,0:1,...])
    phi = 2 * np.pi * sample[:,:,1:2,...]
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1 + Vh[:,:,2:3,...])
    t2 = torch.lerp(sqrt_(1 - t1**2), t2, s)
    Nh = t1 * T1 + t2 * T2 + sqrt_(torch.clamp(1 - t1*t1 - t2*t2, min=0)) * Vh
    h = F.normalize(torch.cat([alpha * Nh[:,:,0:1,...], alpha * Nh[:,:,1:2,...], torch.clamp(Nh[:,:,2:3,...], min=0)], dim=2), dim=2, eps=1e-8)
    wo = reflect(wi, h)

    HdotL = torch.clamp(torch.sum(h*wo, dim=2, keepdim=True), min=0.0001, max=1.0)
    NdotL = torch.clamp(wo[:,:,2:3,...], min=0.0001, max=1.0)
    NdotV = torch.clamp(wi[:,:,2:3,...], min=0.0001, max=1.0)
    # NdotH = torch.clamp(h[:,:,2:3,...], min=0.00001, max=1.0)
    # F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL)
    weight = evalFresnel(Ks, shadowedF90(Ks), HdotL) * sample_weight_ggx(alpha*alpha, NdotL, NdotV) / pS.unsqueeze(1)

    wo = torch.where(sample_diffuse.unsqueeze(2), wo_diff, wo)
    weight = torch.where(sample_diffuse.unsqueeze(2), weight_diff, weight)

    return wo, weight


# 采样 GGX 镜面反射 BRDF。GGX 镜面反射 BRDF 是一个基于微表面理论的 BRDF，它可以模拟真实世界中各种材质的镜面反射特性。
def sample_ggx_specular(sample: torch.Tensor, roughness: torch.Tensor, wi: torch.Tensor):
    """
    :param: sample (bn, spp, 2, h, w)
    :param: roughness (bn, 1, h, w)
    :param: wi (*, *, 3, h, w), supposed to be normalized
    :return: wo (bn, spp, 3, h, w), phi (bn, spp, h, w), cos theta (bn, spp, h, w)
    """
    roughness = roughness.unsqueeze(1)
    alpha = roughness * roughness
    # alpha = roughness
    Vh = F.normalize(torch.cat([alpha * wi[:,:,0:1,...], alpha * wi[:,:,1:2,...], wi[:,:,2:3,...]], dim=2), dim=2, eps=1e-8)
    # bn, spp, _, row, col = Vh.shape
    # Vh = Vh.view(-1, 3, row, col)
    # T1, T2, Vh = utils.hughes_moeller(Vh)
    # T1 = T1.view(bn, spp, 3, row, col)
    # T2 = T2.view(bn, spp, 3, row, col)
    # Vh = Vh.view(bn, spp, 3, row, col)
    lensq = Vh[:,:,0:1,...]**2 + Vh[:,:,1:2,...]**2
    zero_ = torch.zeros_like(Vh[:,:,0,...])
    one_ = torch.ones_like(Vh[:,:,0,...])
    # T1 和 T2 是虚拟半向量 Vh 的两个正交向量。它们用于计算出射光线方向 wo
    T1 = torch.where(
        lensq > 0, 
        torch.stack([-Vh[:,:,1,...], Vh[:,:,0,...], zero_], dim=2) / sqrt_(lensq),
        torch.stack([one_, zero_, zero_], dim=2)
    )
    # cross() 是向量叉积运算
    T2 = torch.cross(Vh, T1, dim=2)
    # 这段代码用于计算采样点的极坐标
    r = sqrt_(sample[:,:,0:1,...])
    phi = 2 * np.pi * sample[:,:,1:2,...]
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1 + Vh[:,:,2:3,...])
    t2 = torch.lerp(sqrt_(1 - t1**2), t2, s)
    Nh = t1 * T1 + t2 * T2 + sqrt_(torch.clamp(1 - t1*t1 - t2*t2, min=0)) * Vh
    # 半向量,半向量是入射光线方向 wi 和出射光线方向 wo 的加权平均值
    h = F.normalize(torch.cat([alpha * Nh[:,:,0:1,...], alpha * Nh[:,:,1:2,...], torch.clamp(Nh[:,:,2:3,...], min=0)], dim=2), dim=2, eps=1e-8)
    wo = reflect(wi, h)
    return wo