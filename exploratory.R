library(caret)
library(dplyr)

setwd("/Users/sdini/PycharmProjects/CropClass/")

data = read.csv("Images/pixels.csv")

data = data %>% filter(cloud_prob <= 2)

table(data$label)

ind.other = which(data$label == "Other")

ind.other.exc = ind.other[sample(length(ind.other) - 15000)]

data.sub = data %>% filter(!(row_number() %in% ind.other.exc))

table(data.sub$label)

featurePlot(x = data[, 1:12], 
            y = data$label, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

transparentTheme(trans = .9)
featurePlot(x = data[, 1:12], 
            y = data$label %>% as.factor(),
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "",
            layout = c(4, 3), 
            auto.key = list(columns = 3)
            )

ggsave("featurePlot.png", width=10, height=5)
