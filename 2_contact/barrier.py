import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)

side_length=0.8
n_seg=10
step=side_length/n_seg
n=(n_seg+1)**2 
damping=0.995
stifness=500.0

y_ground=-0.60
d_hat=0.01
kappa_barrier=1e5
contact_area=1.0
eps_d=1e-8 #prevent divide by 0

x=ti.Vector.field(2,ti.f32,shape=n)
v=ti.Vector.field(2,ti.f32,shape=n)
m=ti.field(ti.f32,shape=n)

H=n_seg*(n_seg+1)
V=H
D=2*n_seg*n_seg
E=H+V+D

edges=ti.Vector.field(2,ti.i32,shape=E)
l2=ti.field(ti.f32,shape=E)
f=ti.Vector.field(2,ti.f32,shape=n)
k=ti.field(ti.f32,shape=E)

is_dbc=ti.field(ti.i32,shape=n)
x_fixed=ti.Vector.field(2,ti.f32,shape=n)
gravity=ti.Vector([0.0,-9.8],ti.f32)

x_tilde=ti.Vector.field(2,ti.f32,shape=n)
grad=ti.Vector.field(2,ti.f32,shape=n)
E_scalar=ti.field(ti.f32,shape=())

max_triplets = 2 * n+32*E   #we just add more DOF here in case it run out of space

@ti.kernel
def init_nodes():
    for i in range (n_seg+1):
        for j in range (n_seg+1):
            index=i*(n_seg+1)+j
            x[index]=[-side_length/2+i*step,-side_length/2+j*step]
            v[index]=ti.Vector([0.0,0.0])
            m[index]=1.0
            is_dbc[index]=0
            x_fixed[index]=x[index]
    #pin the top right and top left nodes
    is_dbc[0*(n_seg+1)+n_seg]=1
    is_dbc[n_seg*(n_seg+1)+n_seg]=1
    


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
        k[e]=stifness

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
        if(is_dbc[i]==0):
            grad[i]+=(-m[i])*gravity

#hessian is 0

@ti.kernel
def x_tilde_val(h:ti.f32):
    for i in range(n):
        x_tilde[i]=x[i]+h*v[i]
        
@ti.kernel
def inertia_val(h:ti.f32):
    for i in range (n):
        diff=x[i]-x_tilde[i]
        ti.atomic_add(E_scalar[None],0.5*m[i]/(h*h)*diff.dot(diff))

@ti.kernel
def inertia_gradient(h:ti.f32):
    for i in range (n):
        if(is_dbc[i]==0):
            grad[i]+=m[i]/(h*h)*(x[i]-x_tilde[i])

@ti.kernel
def inertia_hessian(A: ti.types.sparse_matrix_builder(),h:ti.f32):
    for i in range (n):
        for r in ti.static(range(2)):
            dof=2*i+r
            if(is_dbc[i]==1):
                A[dof,dof]+=1.0
            else:
                A[dof,dof]+=m[i]/(h*h)


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
        if(is_dbc[i]==0):
            ti.atomic_add(grad[i], g_diff)
        if(is_dbc[j]==0):
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
                fi=(is_dbc[i]==1)
                fj=(is_dbc[j]==1)
                #i,i point
                if not fi:
                    A[dof_i_r, dof_i_c] += val
                #i,j and j,i point
                if(not fi) and (not fj):
                    A[dof_i_r, dof_j_c] += -val
                    A[dof_j_r, dof_i_c] += -val
                #j,j point
                if(not fj):
                    A[dof_j_r, dof_j_c] += val
                    
@ti.kernel
def velocity_damping(d:ti.f32):
    for i in range(n):
        v[i]*=d
 
@ti.kernel
def barrier_val():
    for i in range (n):
        d=x[i][1]-y_ground
        if d>0.0 and d<d_hat:
            s=d/d_hat
            ti.atomic_add(E_scalar[None],contact_area*d_hat*kappa_barrier*0.5*(s-1.0)*ti.log(s))

@ti.kernel
def barrier_grad():
    for i in range (n):
       if is_dbc[i]==0:
           d=x[i][1]-y_ground
           if (d>0.0 and d<d_hat):
                s=d/d_hat
                d_safe=ti.max(d,eps_d)
                bp=contact_area*d_hat*kappa_barrier*0.5*(ti.log(s)/d_hat+(s-1)/d_safe)
                grad[i][1] += bp       
    
@ti.kernel
def barrier_hess(A:ti.types.sparse_matrix_builder()):
    for i in range(n):
        d=x[i][1]-y_ground
        if d>0.0 and d<d_hat:
            d_safe=ti.max(d,eps_d)
            bpp=contact_area*(kappa_barrier*0.5)*(d+d_hat)/(d_safe*d_safe)
            dof_y=2*i+1
            if is_dbc[i]==0:
                A[dof_y,dof_y]+=bpp
                
       
                   
#now, we build the solver                
def build_hessian(h):
    A_builder = ti.linalg.SparseMatrixBuilder(
        2*n, 2*n,
        max_num_triplets=max_triplets
    )
    inertia_hessian(A_builder,h)
    mass_spring_hess(A_builder)
    barrier_hess(A_builder)
    A=A_builder.build()
    return A

def compute_grad_and_energy(h):
    clear_grad_energy()
    inertia_val(h)
    inertia_gradient(h)
    gravity_val()
    gravity_grad()
    mass_spring_val()
    mass_spring_grad()
    barrier_val()
    barrier_grad()

def build_gradient():
    b=np.zeros(2*n,dtype=np.float32)
    g=grad.to_numpy()
    for i in range(n):
        if (is_dbc[i]==0):
            b[2*i]=-g[i,0]
            b[2*i+1]=-g[i,1]
        else:
            b[2*i]=0.0
            b[2*i+1]=0.0
    return b

def solve_system(A,b):
    solver=ti.linalg.SparseSolver(solver_type="LDLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    dir=solver.solve(b)
    success=solver.info()
    if not success:
        print("The solver failed")
    return dir

def apply_dir(dir):
    dir=dir.reshape(n,2)
    x_np=x.to_numpy()
    x_np+=dir
    x.from_numpy(x_np)
        
def compute_energy(h):
    clear_grad_energy()
    inertia_val(h)
    gravity_val()
    mass_spring_val()
    barrier_val()
    return E_scalar[None]

def ccd_alpha(x,dir):
    alpha=1.0
    for i in range(n):
        if is_dbc.to_numpy()[i]==1:
            continue
        py=dir[i,1]
        if py<0:
            d=x[i,1]-y_ground
            if d>0:
                alpha_i=0.9*d/(-py)
                alpha=min(alpha,alpha_i)
    return alpha
        
def line_search(dir,h,g0_flat):
    alpha=1.0
    c=1e-4
    
    E0=compute_energy(h)
    dir_flat=dir.reshape(2*n)
    descent=np.dot(g0_flat,dir_flat)
    if descent >= 0:
        print("Not a descent direction")
        return 0.0
    x_backup=x.to_numpy()
    
    alpha = min(alpha, ccd_alpha(x_backup, dir.reshape(n,2)))
    #until it really decreases
    while alpha>1e-6:
        trail=x_backup+alpha*dir.reshape(n,2)
        x.from_numpy(trail)
        
        E_new=compute_energy(h)
        if(E_new<=E0+c*alpha*descent):
            x.from_numpy(x_backup)
            return alpha
        alpha*=0.5
         
    #if it fails to converge, we don't update it    
    x.from_numpy(x_backup)
    return 0.0
        
def newton_step(h):
    compute_grad_and_energy(h)
    g0_flat=grad.to_numpy().reshape(2*n)
    H=build_hessian(h)
    g=build_gradient()
    dir=solve_system(H,g)
    alpha=line_search(dir,h,g0_flat)
    if(alpha==0.0):
        print("Line search fails")
        return            
    apply_dir(alpha*dir)  
        
            
        


@ti.kernel
def perturb():
    x[0] += ti.Vector([0.0, -0.2])
    x[n_seg+1] += ti.Vector([0.0, -0.5])

    
h = 0.01
gui = ti.GUI("Implicit Mass Spring Grid", res=(600, 600))

init_nodes()
init_edge()
init_physics()
#perturb()

while gui.running:

    x_old = x.to_numpy()
    x_tilde_val(h)

    for _ in range(5):
        newton_step(h)

    # update velocity
    x_new = x.to_numpy()
    v.from_numpy((x_new - x_old) / h)
    velocity_damping(damping)

    #rendering
    gui.clear(0x112F41)
    scale=0.7
    offset = np.array([0.0, 0.1])
    pos = x_new*scale + 0.5+offset
    e_np = edges.to_numpy()
    y_ground_screen = y_ground * scale + 0.5 + offset[1]
    gui.line(begin=(0.0, y_ground_screen),
    end=(1.0, y_ground_screen),
    radius=2,
    color=0xFFAA00)

    for e in range(E):
        i = e_np[e][0]
        j = e_np[e][1]
        gui.line(pos[i], pos[j], radius=1, color=0xFFFFFF)

    gui.circles(pos, radius=3, color=0x66CCFF)

    gui.show()
    