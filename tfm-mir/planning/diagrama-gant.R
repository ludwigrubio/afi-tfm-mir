packages <- c("plotly", "RColorBrewer")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}


default.theme <- theme_get()
source("theme_ludwig.R")
#Seteamos nuestor tema
theme_set(theme_ludwig)

# Read in data
df <- read.csv("GanttChart-updated.csv", stringsAsFactors = F)

# Convert to dates
df$Start <- as.Date(df$Start, format = "%m/%d/%Y")

# Choose colors based on number of resources
cols <- RColorBrewer::brewer.pal(length(unique(df$Resource)), name = "Set3")
df$color <- factor(df$Resource, labels = cols)

# Initialize empty plot
p <- plot_ly()

# Each task is a separate trace

for(i in 1:(nrow(df))){
  p <- add_trace(p,
                 x = c(df$Start[i], df$Start[i] + df$Duration[i]),
                 y = c(i, i), 
                 mode = "lines",
                 line = list(color = df$color[i], width = 20),
                 showlegend = F,
                 hoverinfo = "text",
                 
                 # Create custom hover text
                 text = paste("Tarea: ", df$Task[i], "<br>",
                              "Duracion: ", df$Duration[i], " días<br>"
                              ),
                 
                 evaluate = F
  )
}


# Add information to plot and make the chart more presentable
p <- layout(p,
            
            xaxis = list(
              showgrid = F,
              tickformat = "%d-%m",
              tickangle = 45,
              range = c("2019-05-10","2019-06-26")
            ),
            
            yaxis = list(tickfont = list(color = "#333333"),
                         tickmode = "array", tickvals = 1:nrow(df), ticktext = unique(df$Task),
                         domain = c(0, 0.8)),
            
            # Annotations
            annotations = list(

              list(xref = "paper", yref = "paper",
                   x = 0.80, y = 0.87,
                   text = paste0("Duración total: ", sum(40 * 4), " horas"),
                   font = list(color = "#aaaaaa", size = 12),
                   ax = 0, ay = 0,
                   align = "left"),
              
              # Add title
              list(xref = "paper", yref = "paper",
                   x = 0.1, y = 1, xanchor = "left",
                   text = paste0("Diagrama de Grantt - TFM"),
                   font = list(color = "#000000", size = 20),
                   ax = 0, ay = 0,
                   align = "left")
            ))

p
theme_set(default.theme)
