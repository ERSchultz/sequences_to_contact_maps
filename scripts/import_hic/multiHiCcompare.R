library(multiHiCcompare)
library(dplyr)
chrlens = c(249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663,
            146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540,
            102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566,
            155270560, 59373566)

# https://github.com/lhqxinghun/bioinformatics/blob/master/Hi-C/NormCompare/sparse2matrix.R
#function name:sparse2matrix
#function:transform data from sparse format to matrix format
#parameters:
#           sparse:the sparse data
#           dim:the dimension of matrix
#           resolution:the resolution of input data
sparse2matrix <- function(sparse, dim, resolution) 
{
  sparsetriangular <- sparse;
  if (ncol(sparsetriangular) >= 3) 
  {
    bins <- unique(c(sparsetriangular[, 1], sparsetriangular[, 2]))
    bins <- as.numeric(bins)
    bins <- bins[order(bins)]
    bins <- unique(bins)
    bin.size <- min(diff(bins))
    if(bin.size != resolution)
    {
      stop("Resolution assigned by user is not compatible with the corrodinates of bins")       
    }
    hicmatrix <- matrix(0, dim, dim)
    rownum <- sparsetriangular[, 1] / bin.size + 1
    colnum <- sparsetriangular[, 2] / bin.size + 1
    hicmatrix[cbind(rownum, colnum)] <- as.numeric(c(sparsetriangular[, 3]))
    hicmatrix[cbind(colnum, rownum)] <- as.numeric(c(sparsetriangular[, 3]))
    hicmatrix[is.na(hicmatrix)] <- 0
    
    return(hicmatrix)
  }
  else
  {
    stop("The number of columns of a sparse matrix should be not less than 3")       
  }
}

dataset='11_20_23'
cell_line_list <- list()
replicates=9
data_dir=sprintf('/home/erschultz/dataset_%s', dataset)
# load data in sparse upper triangular format1
for (chr in 1:22) {
  for (i in 0:(replicates-1)) {
    file = sprintf("%s/chroms_rep%s/chr%s/y_sparse.txt", data_dir, i, chr)
    print(file)
    table = read.table(file)
    table$V1 = as.factor(table$V1)
    cell_line_list[[i+1]] = data.frame(table)
  }
  # make groups & covariate input
  groups <- factor(c(rep(1, length(cell_line_list))))
  
  # make the hicexp object
  hicexp <- make_hicexp(data_list = cell_line_list, groups = groups, remove.regions=NULL, remove_zeros=FALSE, filter=FALSE)
  hicexp <- fastlo(hicexp)
  
  
  hic <- data.frame(hicexp@hic_table)
  cols = paste("IF", seq(1,replicates), sep='')
  hic_combined <- hic %>%
    mutate(sum = rowSums(across(all_of(cols)))) %>%
    select(c('region1', 'region2', 'sum'))
  resolution=50000
  dim <- floor(chrlens[chr] / resolution) + 1
  hic_full <- sparse2matrix(hic_combined, dim, resolution)
  write.table(round(hic_full, digits=6), sprintf("%s/chr%s_multiHiCcompare.txt", data_dir, chr), row.names = FALSE, col.names = FALSE, sep = "\t")
  for (i in 0:(replicates-1)){
    dir=sprintf("%s/chroms_rep%s/chr%s/y_multiHiCcompare.txt", data_dir, i, chr)
    hicsparsenorm <- data.frame(subset(hic,select=c(2,3,i+5)))
    hic_full <- sparse2matrix(hicsparsenorm, dim, resolution)
    write.table(round(hic_full, digits=6), dir, row.names = FALSE, col.names = FALSE, sep = "\t")
  }
}
