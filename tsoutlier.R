library(tsoutliers)

mydata <- read.csv('./data/data2.csv', sep=";")

y <- mydata['val']
x <- mydata['offset']
label <- mydata['error']

fit <- arima(y[1:12000,], order = c(1, 1, 1))
resid <- residuals(fit)
pars <- coefs2poly(fit)
otypes <- c("AO")
print("Calculating outliers")
mo0 <- locate.outliers(resid, pars, types = otypes)
outliers <- mo0['ind']
outliersA <- data.matrix(outliers)


TP = 0;
FP = 0;
TN = 0;
FN = 0;

for (i in 1:nrow(label)) {
  PP = 0;
  RP = 0;
  
  if (i %in% outliersA) PP = 1;
  if (label[i, "error"] == "True") RP = 1;
  
  if ((RP == 1) && (PP == 1)) TP = TP + 1
  if ((RP == 0) && (PP == 0)) TN = TN + 1
  if ((RP == 1) && (PP == 0)) FN = FN + 1
  if ((RP == 0) && (PP == 1)) FP = FP + 1
  #print(label[outlier, "error"])
}

cat(sprintf("TP %d, TN %d, FP %d, FN %d", TP, TN, FP, FN));

prec = TP / (TP + FP)
rec = TP / (TP + FN)

F1 = 2 * prec * rec / (prec + rec)


cat(sprintf("\n %.3f & %.3f & %.3f", prec, rec, F1))

