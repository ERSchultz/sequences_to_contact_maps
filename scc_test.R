dir = '/home/erschultz'

ifile = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017/samples/1CDS2.1/chrom10.txt'
# ifile = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples/sample3/y.txt'
mat1 = read.table(ifile)

ifile = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017/samples/1CDS2.2/chrom10.txt'
# ifile = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples/sample3/PCA-normalize/k4/replicate1/y.txt'
mat2 = read.table(ifile)

scc.out = get.scc(mat1, mat2, resol = 50000, h = 2, ubr = 500000)
scc.out

for (x in 0:3) {
  diag1 <- unlist(mat1[row(mat1) == (col(mat1) - x)])
  diag2 <- unlist(mat2[row(mat2) == (col(mat2) - x)])
  p <- cor(diag1, diag2, method = 'pearson')
  # print(diag1)
  # print(diag2)
  print(p)
}


my.get.scc <- function (mat1, mat2, resol, h, lbr = 0, ubr = 5000000){
  
  if (h == 0){
    smt_R1 = mat1
    smt_R2 = mat2
  } else {
    smt_R1 = fastMeanFilter(as.matrix(mat1), h)
    rm(mat1)
    smt_R2 = fastMeanFilter(as.matrix(mat2), h)
    rm(mat2)
  }
  
  lb <- floor(lbr/resol)
  ub <- floor(ubr/resol)
  corr <- array(ub-lb+1)
  cov <- array(ub-lb+1)
  wei <- array(ub-lb+1)
  n <- array(ub-lb+1)
  
  est.scc = function(dist){
    
    ffd1 <- ffd2 <- NULL
    for (i in 1:(ncol(smt_R1)-dist)){
      
      ffd1 <- c(ffd1, smt_R1[i+dist, i])
      ffd2 <- c(ffd2, smt_R2[i+dist, i])
      # filt <- which(ffd1 == 0 & ffd2 == 0)
      filt <- which(abs(ffd1) < 1e-12 & abs(ffd2) < 1e-12)
      if (length(filt) == 0){
        ffd <- cbind(ffd1, ffd2)
      } else
        ffd <- cbind(ffd1[-filt], ffd2[-filt])
    }
    
    if (nrow(ffd) != 0){
      n = nrow(ffd)
      nd = vstran(ffd)
      
      # print(n)
      # print(ffd)
      
      if (length(unique(ffd[,1])) != 1 
          & length(unique(ffd[,2])) != 1) {
        corr = cor(ffd[,1], ffd[,2])
        cov = cov(nd[,1], nd[,2])
        wei = sqrt(var(nd[,1])*var(nd[,2]))*n
      } else {
        corr = NA
        cov = NA
        wei = NA
      }
    } else {
      corr = NA 
      cov = NA
      wei = NA
    }
    
    return(list(corr = corr, wei = wei))
  }
  
  st = sapply(seq(lb,ub), est.scc)
  corr0 = unlist(st[1,])
  wei0 = unlist(st[2,])
  
  corr = corr0[!is.na(corr0)]
  wei = wei0[!is.na(wei0)]
  scc = corr %*% wei/sum(wei)
  std = sqrt(sum(wei^2*var(corr))/(sum(wei))^2)
  
  return(list(corr = corr, wei = wei, scc = scc, std = std))
}
