import taichi as ti
import math

ti.init(arch=ti.cpu)

side_length=1.0
n_seg=10
step=side_length/n_seg
n=(n_seg+1)**2 #121 points in total

x=ti.Vector.field(2,ti.f32,shape=n)
v=ti.Vector.field(2,ti.f32,shape=n)
m=ti.field(ti.f32,shape=n)

H=n_seg*(n_seg+1)
V=H
D=2*n_seg*n_seg
E=H+V+D

edges=ti.Vector.field(2,ti.i32,shape=E) #edges是整数,i32
l2=ti.field(ti.f32,shape=E)#对于每一个弹簧，都有stifness 
f=ti.Vector.field(2,ti.f32,shape=n)
k=ti.field(ti.f32,shape=E)

is_dbc=ti.field(ti.i32,shape=n)
gravity=ti.Vector([0.0,-9.81],ti.f32)

x_tilde=ti.Vector.field(2,ti.f32,shape=n)
grad=ti.Vector.field(2,ti.f32,shape=n)
E_scalar=ti.field(ti.f32,shape=())

max_triplets = 2 * n+16*E   # 先只装 inertia：每个点 2 个对角项，然后16E
A_builder = ti.linalg.SparseMatrixBuilder(2 * n, 2 * n, max_num_triplets=max_triplets)




@ti.kernel
def init_nodes():
    for i in range (n_seg+1):
        for j in range (n_seg+1):
            index=i*(n_seg+1)+j
            x[index]=[-side_length/2+i*step,-side_length/2+j*step]
            v[index]=ti.Vector([0.0,0.0])
            m[index]=1.0
            is_dbc[index]=0


@ti.kernel
def init_edge():
    #horizontal
    for i in range (n_seg):
        for j in range(n_seg+1):
            edge_id=i*(n_seg+1)+j
            edges[edge_id]=[i*(n_seg+1)+j,(i+1)*(n_seg+1)+j]
    #vertical                    
    for i in range (n_seg+1):
        for j in range(n_seg):
            edge_id=H+i*n_seg+j
            edges[edge_id]=[i*(n_seg+1)+j,i*(n_seg+1)+j+1]
    base=H+V
    #diagonals
    for i in range (n_seg):
        for j in range (n_seg):
            cell_id=i*n_seg+j #一共有n_seg**2个格子，每个格子要写两条边.所以*2
            id0=base+cell_id*2 
            id1=base+cell_id*2+1     
            edges[id0]=[i*(n_seg+1)+j,(i+1)*(n_seg+1)+j+1]
            edges[id1]=[(i+1)*(n_seg+1)+j,i*(n_seg+1)+j+1]

@ti.kernel
def init_physics():
    for e in range (E):
        i=edges[e][0]
        j=edges[e][1]
        diff=x[i]-x[j]
        l2[e]=diff.dot(diff)
        k[e]=500.0

@ti.kernel
def clear_grad_energy():
    for i in range(n):
        grad[i]=ti.Vector([0.0,0.0])
    E_scalar[None]=0.0

@ti.kernel
def gravity_val():
    for i in range(n):
        ti.atomic_add(E_scalar[None],-m[i]*x[i].dot(gravity))

@ti.kernel
def gravity_grad():
    for i in range (n):
        grad[i]+=(-m[i])*gravity

#hessian is 0

@ti.kernel
def x_tilde_val(h:ti.f32):
    for i in range(n):
        x_tilde[i]=x[i]+h*v[i]
        
@ti.kernel
def inertia_val():
    for i in range (n):
        diff=x[i]-x_tilde[i]
        ti.atomic_add(E_scalar[None],0.5*m[i]*diff.dot(diff))

@ti.kernel
def inertia_gradient():
    for i in range (n):
        grad[i]+=m[i]*(x[i]-x_tilde[i])

@ti.kernel
def inertia_hessian(A: ti.types.sparse_matrix_builder()):
    for i in range (n):
        for r in ti.static(range(2)):
            dof=2*i+r
            A[dof,dof]+=m[i]


@ti.kernel
def mass_spring_val():
    for e in range(E):
        i=edges[e][0]
        j=edges[e][1]
        diff=x[i]-x[j]
        s=diff.dot(diff)/l2[e]-1
        energy=0.5*k[e]*l2[e]*s*s
        ti.atomic_add(E_scalar[None],energy)
    
    
    
@ti.kernel
def mass_spring_grad():
    for e in range(E):
        i=edges[e][0]
        j=edges[e][1]
        diff=x[i]-x[j]
        s=diff.dot(diff)/l2[e]-1
        g_diff=2*k[e]*s*diff
        ti.atomic_add(grad[i], g_diff)
        ti.atomic_add(grad[j], -g_diff)
        
    
@ti.kernel
def mass_spring_hess(A: ti.types.sparse_matrix_builder()):
    for e in range(E):
        i=edges[e][0]
        j=edges[e][1]
        diff=x[i]-x[j]
        dx=diff[0]
        dy=diff[1]
        dd=dx*dx+dy*dy
        coeff=2*k[e]/l2[e]
        #the four entries of unit matrix H
        H00=coeff*(2.0*dx*dx+(dd-l2[e]))
        H01=coeff*(2.0*dx*dy)
        H10=H01
        H11=coeff*(2.0*dy*dy+(dd-l2[e]))
        H_local=ti.Matrix([[H00,H01],
                          [H10,H11]])
        for r in ti.static(range(2)):
            for c in ti.static(range(2)):
                dof_i_r=2*i+r
                dof_i_c=2*i+c
                dof_j_r=2*j+r
                dof_j_c=2*j+c
                val=H_local[r,c]
                A[dof_i_r, dof_i_c] += val
                A[dof_i_r, dof_j_c] += -val
                A[dof_j_r, dof_i_c] += -val
                A[dof_j_r, dof_j_c] += val
                
        
        
        


@ti.kernel
def compute_force():
    for i in range (n):
        f[i]=ti.Vector([0.0,0.0])
    for e in range (E):
        i=edges[e][0]
        j=edges[e][1]
        diff=x[i]-x[j]
        s=diff.dot(diff)/l2[e]-1
        f_spring=2*k[e]*s*diff
        ti.atomic_add(f[i],-f_spring)
        ti.atomic_add(f[j],f_spring)
    #for i in range(n):
        #f[i]+=ti.Vector([0.0,-9.8])*m[i]

    
@ti.kernel
def explicit_step(h:ti.f32):
    for i in range(n):
        v[i]+=h*f[i]/m[i]
        v[i]*=0.999
        x[i]+=h*v[i]

@ti.kernel
def perturb():
    x[0] += ti.Vector([0.0, -0.2])

gui = ti.GUI("Mass Spring Grid", res=(600, 600))
init_nodes()
init_edge()
init_physics()
perturb()
while gui.running:
    
    compute_force()
    explicit_step(0.005)
    gui.clear(0x112F41)
    pos = x.to_numpy() + 0.5          # [-0.5,0.5] -> [0,1]
    e_np = edges.to_numpy()
    # 画弹簧
    for e in range(E):
        i = e_np[e][0]
        j = e_np[e][1]
        gui.line(pos[i], pos[j], radius=1, color=0xFFFFFF)
    # 画点
    gui.circles(pos, radius=3, color=0x66CCFF)
    gui.show()




















