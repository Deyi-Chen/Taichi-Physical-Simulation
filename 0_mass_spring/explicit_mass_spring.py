import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

side_length=0.8
n_seg=10
step=side_length/n_seg
n=(n_seg+1)**2 
stiffness=500

x=ti.Vector.field(2,ti.f32,shape=n)
v=ti.Vector.field(2,ti.f32,shape=n)
f=ti.Vector.field(2,ti.f32,shape=n)
m=ti.field(ti.f32,shape=n) 
E=n_seg*(n_seg+1)*2+2*n_seg*n_seg #number of springs
edges=ti.Vector.field(2,ti.i32,shape=E) 
l2=ti.field(ti.f32,shape=E)
k=ti.field(ti.f32,shape=E)

@ti.kernel
def init_nodes():
    for i in range (n_seg+1):
        for j in range (n_seg+1):
            x[i*(n_seg+1)+j]=[-side_length/2+i*step,-side_length/2+j*step]
            
@ti.kernel
def init_edge():
    H=n_seg*(n_seg+1)
    V=H
    #along the i direction
    for i in range (n_seg):
        for j in range(n_seg+1):
            edge_id=i*(n_seg+1)+j
            edges[edge_id]=[i*(n_seg+1)+j,(i+1)*(n_seg+1)+j]
    #along the j direction                    
    for i in range (n_seg+1):
        for j in range(n_seg):
            edge_id=H+i*n_seg+j
            edges[edge_id]=[i*(n_seg+1)+j,i*(n_seg+1)+j+1]
    base=H+V
    #diagonal
    for i in range (n_seg):
        for j in range (n_seg):
            cell_id=i*n_seg+j
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
        k[e]=stiffness
    for i in range(n):
        m[i]=1.0
        v[i]=ti.Vector([0.0,0.0]) 
        
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

@ti.kernel
def explicit_step(h:ti.f32):
    for i in range(n):
        v[i]+=h*f[i]/m[i]
        v[i]*=0.999 #damping
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
    pos = x.to_numpy() + 0.5          
    e_np = edges.to_numpy()
    for e in range(E):
        i = e_np[e][0]
        j = e_np[e][1]
        gui.line(pos[i], pos[j], radius=1, color=0xFFFFFF)
        
    gui.circles(pos, radius=3, color=0x66CCFF)
    gui.show()


