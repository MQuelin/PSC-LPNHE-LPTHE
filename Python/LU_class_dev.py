class TestLULayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lu_decomposed=True):
        super().__init__()
        # w = special_ortho_group.rvs(dim)
        # print(w)
        w = np.random.rand(output_dim, input_dim)
        w = torch.from_numpy(w)
        print(w)
        if lu_decomposed:
            p, l, u = torch.linalg.lu(w)
            self.p = p
            self.l = l
            self.u = u
        else:
            self.w = nn.Parameter(torch.from_numpy(w).float().cuda())
        self.lu_decomposed = lu_decomposed
    
    @staticmethod
    def compose_w(p, l, u):
        l = torch.tril(l)
        u = torch.triu(u)
        return torch.mm(torch.mm(p, l), u)
  
    def forward(self, x, reverse="false"):
        print("Bloop")
        if not reverse :
            if self.lu_decomposed:
                w = self.compose_w(self.p, self.l, self.u)
            else:
                w = self.w
            y = torch.mm(x, w)
            return y, torch.log(torch.abs(torch.det(w)))
        else :
            return self.invert(x)
    
    def invert(self, y):
        if self.lu_decomposed:
            w = self.compose_w(self.p, self.l, self.u)
            log_det = torch.sum(torch.log(
                torch.abs(torch.diagonal(self.u))))
        else:
            w = self.w
            log_det = torch.log(
                torch.abs(torch.det(w)))
        x = torch.mm(y, torch.inverse(w))
        return x, log_det.expand(x.shape[0])
