import torch
from e8_matrix_lattice_fast import E8MatrixLattice


def test_embedded_m3_reps():
    P=E8MatrixLattice(dtype=torch.float64)
    assert P.A3.shape==(2240,8,8)
    I=torch.eye(8,dtype=torch.float64)
    err=(P.A3.transpose(-1,-2)@P.A3-3*I).abs().max()
    assert float(err)==0.0


def test_m3_projector_returns_lattice_shell():
    torch.manual_seed(1)
    P=E8MatrixLattice(dtype=torch.float64)
    X=torch.randn(2,8,8,dtype=torch.float64)
    M,d=P.project_m3(X,return_distance=True)
    assert bool(P.is_lattice_point(M,atol=1e-9).all())
    assert float(P.shell_residual(M,3).max()) < 1e-10
    direct=(X-M).square().sum((-2,-1))
    assert torch.allclose(d,direct,atol=1e-9,rtol=1e-9)


def test_m3_known_point_is_fixed():
    P=E8MatrixLattice(dtype=torch.float64)
    X=P.A3[:3].clone()
    M,d=P.project_m3(X,return_distance=True)
    assert torch.allclose(M,X,atol=1e-10,rtol=0)
    assert torch.allclose(d,torch.zeros_like(d),atol=1e-10,rtol=0)


def test_m2_regression():
    torch.manual_seed(2)
    P=E8MatrixLattice(dtype=torch.float64,preload_m2=True)
    X=torch.randn(2,8,8,dtype=torch.float64)
    a,da=P.project_m2(X,algorithm='branch_bound',return_distance=True)
    b,db=P.project_m2(X,algorithm='exhaustive',return_distance=True,batch_chunk_size=1,transform_chunk=512)
    assert torch.allclose(da,db,atol=1e-9,rtol=1e-9)
    assert torch.allclose(a,b,atol=1e-9,rtol=0)

def test_m4_known_product_is_fixed():
    P=E8MatrixLattice(dtype=torch.float64,preload_m2=True)
    X=(P.A2[0]@P.A2[1]).unsqueeze(0)
    M,d=P.project_m4(X,return_distance=True,outer_a_chunk=2)
    assert torch.allclose(M,X,atol=1e-10,rtol=0)
    assert torch.allclose(d,torch.zeros_like(d),atol=1e-10,rtol=0)
    assert float(P.shell_residual(M,4).max()) < 1e-10
