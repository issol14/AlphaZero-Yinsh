import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import os

def draw_yinsh(board_state, save_path='./board.png'):
    fig, ax = plt.subplots(figsize=(6,6)
)
    patches_list = []

    # 1) 삼각 격자 패치 생성 
    for tri in precomputed_triangles:          # (x,y) 3점 좌표 튜플
        poly = patches.Polygon(tri, closed=True,
                               edgecolor='grey', facecolor='none', lw=0.4)
        patches_list.append(poly)

    # 2) 링·마커 그리기
    for (q,r), piece in board_state.items():
        x,y = axial_to_cart(q,r)               # 좌표 변환
        if piece == 'WRING':
            ax.add_patch(patches.Circle((x,y), 0.28, lw=2, fill=False, color='black'))
        elif piece == 'BMARK':
            ax.add_patch(patches.Circle((x,y), 0.2, color='black'))

    ax.add_collection(PatchCollection(patches_list, match_original=True))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 저장 전에 디렉토리 확인
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
