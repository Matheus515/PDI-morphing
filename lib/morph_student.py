import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    return (pA + pB)*0.5
    #raise NotImplementedError("Implemente: pontos_medios")

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    medios = pontos_medios(pA, pB)
    return indices_delaunay(medios)

    #raise NotImplementedError("Implemente: indices_pontos_medios")

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    return a*t + b
    #raise NotImplementedError("Implemente: linear")

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """
    return 1.0 /(1.0 + np.exp(-k * (t - 0.5)))

    #raise NotImplementedError("Implemente: sigmoide")

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    return t

    #raise NotImplementedError("Implemente: dummy")

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]

    return ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    
    #raise NotImplementedError("Implemente: _det3")

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    v1, v2, v3 = tri[0], tri[1], tri[2]
    det_total = _det3(v1, v2, v3)
    if np.isclose(det_total, 0):
        return -1.0, -1.0, -1.0
    
    w1 = _det3(pt, v2, v3) / det_total
    w2 = _det3(v1, pt, v3) / det_total
    w3 = 1.0 - w1 - w2

    return w1, w2, w3

    #raise NotImplementedError("Implemente: _transf_baricentrica")

def _check_bari(w1, w2, w3, eps=1e-6):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """

    return (w1 >= -eps) and (w2 >= -eps) and (w3 >= -eps)

    #raise NotImplementedError("Implemente: _check_bari")

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """

    xmin = np.floor(np.min(tri[:, 0]))
    xmax = np.ceil(np.max(tri[:, 0]))
    ymin = np.floor(np.min(tri[:, 1]))
    ymax =np.ceil(np.max(tri[:, 1]))

    xmin = int(np.clip(xmin, 0, W - 1))
    xmax = int(np.clip(xmax, 0, W - 1))
    ymin= int(np.clip(ymin, 0, H - 1))
    ymax = int(np.clip(ymax, 0, H - 1))

    return xmin, xmax, ymin, ymax

    #raise NotImplementedError("Implemente: _tri_bbox")

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """

    H, W = img_float.shape[:2]
    x = np.clip(x, 0, W - 1.000001)
    y = np.clip(y, 0, H - 1.000001)

    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, W - 1)
    y2 = min(y1 + 1, H - 1)

    a1 = img_float[y1, x1]
    a2 = img_float[y1, x2]
    a3 = img_float[y2, x1]
    a4 = img_float[y2, x2]

    dc = x - x1
    dl = y-y1

    p = a1 * (1-dc)*(1-dl) + a2 * dc*(1-dl) + a3 * (1-dc)*dl + a4 * dc*dl
    return p

    #raise NotImplementedError("Implemente: _amostra_bilinear")

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """

    H, W = A.shape[:2]
    frame_resultante = np.zeros_like(A)
    for t in triangles:
        triA = pA[t]
        triB = pB[t]
        triT = (1.0 - alfa) * triA + alfa * triB
        xmin, xmax, ymin, ymax = _tri_bbox(triT, W, H)

        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                pt = (px, py)
                w1, w2, w3 = _transf_baricentrica(pt, triT)

                if _check_bari(w1, w2, w3):
                    p_em_A = w1 * triA[0] + w2 * triA[1] + w3 * triA[2]
                    p_em_B = w1 * triB[0] + w2 * triB[1] + w3 * triB[2]

                    corA = _amostra_bilinear(A, p_em_A[0], p_em_A[1])
                    corB= _amostra_bilinear(B, p_em_B[0], p_em_B[1])

                    cor_final= (1.0 - beta) * corA + beta * corB
                    frame_resultante[py, px] = cor_final

    return frame_resultante
                
    #raise NotImplementedError("Implemente: gera_frame")
