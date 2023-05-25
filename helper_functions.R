

get_dataset_overview <- function(df){
  
  # Get the number of missing values per column
  missing_values <- sapply(df, function(x) sum(is.na(x)))
  
  # Get the data types of each column
  data_types <- sapply(df, class)
  
  # Get the number of unique values per column
  unique_values <- sapply(df, function(x) length(unique(x)))
  
  # Combine the information into a data frame
  overview <- data.frame(Column = names(df),
                         Perc_missing = 100*(missing_values/nrow(df)),
                         Data_Type = data_types,
                         Unique_Values = unique_values) 
  
  return(overview)
}

clean_read_csv <- function(path){
  df <- read_csv(path, show_col_types = FALSE) %>%
      janitor::clean_names() %>%
      mutate_if(is.character, ~ ifelse(. %in% c('XNA', 'NA', 'unknown', 'Unknown', 'other',
                                                'Other','Other_A','Other_B', 'Others'), NA, .))
  
  
  overview <- get_dataset_overview(df)
  
  ##convert binary cols to factor
  binary_cols <- filter(overview, Unique_Values ==2)$Column
  df[binary_cols] <- lapply(df[binary_cols], factor)
  
  #fill in NAs in categorical columns to "other"
  categ_col <- filter(overview, Data_Type=='character')
  categ_col_w_missing_data <- filter(categ_col, Data_Type=='character')$Column
  is_missing = is.na(df[,categ_col_w_missing_data])
  df[,categ_col_w_missing_data][is_missing] = 'other'
  
  ##now convert all categorical columns to factors
  df[categ_col$Column] <- lapply(df[categ_col$Column], factor)
  
  return(df)
  
  
  
  
}


plot_numeric_variable <- function(feature, df){
  df_to_plot <- df[,c('target',feature)] 
  df_to_plot$target <- ifelse(df_to_plot$target==0, 'Loan Repayed', 'Loan Defaulted')
  colnames(df_to_plot)[colnames(df_to_plot)==feature] <- 'feature_to_plot'
  
  x <- df_to_plot$feature_to_plot[!is.na(df_to_plot$feature_to_plot)]
  x_min <- quantile(x, 0.01)
  x_max <- quantile(x, 0.99)
  
  
  plot <- ggplot(df_to_plot, aes(feature_to_plot, fill = target, colour = target)) +
    geom_density(alpha = 0.25)+
    labs(x=feature,y='Density')+
    scale_color_manual(values = c('Loan Repayed' = 'green',
                                  'Loan Defaulted' = 'red'))+
    scale_fill_manual(values = c('Loan Repayed' = 'green',
                                 'Loan Defaulted' = 'red'))+
    xlim(c(x_min, x_max))+
    theme(legend.position = 'bottom', 
          legend.title = element_blank())
  
  print(plot)
  
  
}


plot_categorical_variable <- function(feature, df){
  df_to_plot <- df 
  colnames(df_to_plot)[colnames(df_to_plot) == feature] <- 'feature_to_plot'  # Renaming the column for plotting
  overall_prop <- df$target %>% as.character %>% as.numeric %>% mean  # Computing overall proportion
  df_to_plot <- df_to_plot %>% 
    group_by(feature_to_plot) %>% 
    summarise(prop_defult = target %>% as.character %>% as.numeric %>% mean, 
              count = n(), 
              effect_dir = ifelse(prop_defult > overall_prop, 'up', 'down'))  # Grouping and summarizing data
  
  arrow_x <- length(unique(df_to_plot$feature_to_plot)) + 0.3  # Positioning for arrow labels
  
  # Adjusting axis angle and hjust based on the number of groups
  if (arrow_x > 4) {
    x_axis_angle <- 35
    x_axis_hjust <- 1
  } else {
    x_axis_angle <- 0
    x_axis_hjust <- 0.5
  }
  
  # Creating the plot
  plot <- ggplot(df_to_plot, aes(x = reorder(feature_to_plot, prop_defult), y = prop_defult)) +
    geom_point(size = 2) +  # Adding point markers
    labs(x = feature, y = 'Proportion of defaulting loans', 
         title = paste('Proportion of Loan defaults by', feature)) +  # Setting axis labels and plot title
    geom_hline(yintercept = overall_prop, linetype = 'dotted') +  # Adding a horizontal dotted line
    geom_segment(aes(x = feature_to_plot, y = prop_defult,
                     xend = feature_to_plot, yend = overall_prop, 
                     color = effect_dir),
                 linewidth = 1.4) +  # Adding segments
    scale_color_manual(values = c('down' = 'green', 'up' = 'red')) +  # Setting color scale for segments
    annotate("curve", x = arrow_x, y = overall_prop, 
             xend = arrow_x, yend = overall_prop - 0.05,
             curvature = 0, arrow = arrow(length = unit(0.03, "npc")),
             color = "green") +  # Adding a green curved arrow
    geom_text(aes(x = arrow_x + 0.1, y = overall_prop - 0.025, 
                  label = "Decreased risk of default", angle = 90), 
              color = "green", size = 2.8) +  # Adding green arrow label
    geom_text(aes(x = arrow_x + 0.1, y = overall_prop + 0.025, 
                  label = "Increased risk of default", angle = 90), 
              color = "red", size = 2.8) +  # Adding red arrow label
    annotate("curve", x = arrow_x, y = overall_prop, 
             xend = arrow_x, yend = overall_prop + 0.05,
             curvature = 0, arrow = arrow(length = unit(0.03, "npc")),
             color = "red") +  # Adding a red curved arrow
    geom_text(aes(x = feature_to_plot, y = prop_defult, label = paste("n =", count)),
              vjust = -0.5, color = "black") +  # Adding count labels
    theme_bw() +  # Setting plot theme +
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5, face = 'bold', size = 13.5),
          axis.text.x = element_text(angle = x_axis_angle, hjust = x_axis_hjust))
  
  print(plot)  # Displaying the plot
  
}