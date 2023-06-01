# Define a function to get an overview of a dataset
get_dataset_overview <- function(df) {
  # Get the number of missing values per column
  missing_values <- sapply(df, function(x) sum(is.na(x)))

  # Get the data types of each column
  data_types <- sapply(df, class)

  # Get the number of unique values per column
  unique_values <- sapply(df, function(x) length(unique(x)))

  # Combine the information into a data frame
  overview <- data.frame(
    Column = names(df),
    Perc_missing = 100 * (missing_values / nrow(df)),
    Data_Type = data_types,
    Unique_Values = unique_values
  )

  return(overview)
}

# Define a function to read and clean a CSV file
clean_read_csv <- function(path) {
  # Read the CSV file and clean the column names
  df <- read_csv(path, show_col_types = FALSE) %>%
    janitor::clean_names()

  # Replace certain values with NA in character columns
  df <- df %>%
    mutate_if(
      is.character,
      ~ifelse(. %in% c('XNA', 'NA', 'unknown', 'Unknown', 'other', 'Other', 'Other_A', 'Other_B', 'Others'), NA, .)
    )

  # Get the dataset overview
  overview <- get_dataset_overview(df)

  # Convert binary columns to factors
  binary_cols <- filter(overview, Unique_Values == 2)$Column
  df[binary_cols] <- lapply(df[binary_cols], factor)

  # Fill in NAs in categorical columns with "other"
  categ_col_w_missing_data <- filter(overview, Data_Type == 'character')$Column
  is_missing <- is.na(df[, categ_col_w_missing_data])
  df[, categ_col_w_missing_data][is_missing] <- 'other'

  # Convert all categorical columns to factors
  df[categ_col_w_missing_data] <- lapply(df[categ_col_w_missing_data], factor)

  return(df)
}

# Define a function to plot a density plot for a numeric variable
plot_numeric_variable <- function(feature, df) {
  # Create a dataframe for plotting
  df_to_plot <- df[, c('target', feature)] %>%
    mutate(target = ifelse(target==0, 'Loan Repayed','Loan Defaulted' ))
  colnames(df_to_plot)[colnames(df_to_plot) == feature] <- 'feature_to_plot'

  # Calculate the range for x-axis
  x <- df_to_plot$feature_to_plot[!is.na(df_to_plot$feature_to_plot)]
  x_min <- quantile(x, 0.01)
  x_max <- quantile(x, 0.99)

  # Create the density plot
  plot <- ggplot(df_to_plot, aes(feature_to_plot, fill = target, colour = target)) +
    geom_density(alpha = 0.25) +
    labs(x = feature, y = 'Density') +
    scale_color_manual(values = c('Loan Repayed' = 'green', 'Loan Defaulted' = 'red')) +
    scale_fill_manual(values = c('Loan Repayed' = 'green', 'Loan Defaulted' = 'red')) +
    xlim(c(x_min, x_max)) +
    theme(legend.position = 'bottom', legend.title = element_blank())

  print(plot)
}

# Define a function to plot a bar chart for a categorical variable
plot_categorical_variable <- function(feature, df) {
  # Create a dataframe for plotting
  df_to_plot <- df
  colnames(df_to_plot)[colnames(df_to_plot) == feature] <- 'feature_to_plot'  # Renaming the column for plotting
  overall_prop <- df$target %>% as.character %>% as.numeric %>% mean  # Computing overall proportion
  df_to_plot <- df_to_plot %>%
    group_by(feature_to_plot) %>%
    summarise(prop_default = mean(target %>% as.character %>% as.numeric),
              count = n(),
              # Grouping and summarizing data
              effect_dir = ifelse(prop_default > overall_prop, 'up', 'down'))

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
  plot <- ggplot(df_to_plot,
                 aes(x = reorder(feature_to_plot, prop_default),
                     y = prop_default)) +
    geom_point(size = 2) +  # Adding point markers
    # Setting axis labels and plot title
    labs(x = feature, y = 'Proportion of defaulting loans',
         title = paste('Proportion of Loan defaults by', feature)) +
    # Adding a horizontal dotted line
    geom_hline(yintercept = overall_prop, linetype = 'dotted') +
    geom_segment(aes(x = feature_to_plot, y = prop_default,
                     xend = feature_to_plot, yend = overall_prop,
                     color = effect_dir),
                 linewidth = 1.4) +  # Adding segments
    # Setting color scale for segments
    scale_color_manual(values = c('down' = 'green', 'up' = 'red')) +
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
    geom_text(aes(x = feature_to_plot, y = prop_default, label = paste("n =", count)),
              vjust = -0.5, color = "black") +  # Adding count labels
    theme_bw() +  # Setting plot
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5, face = 'bold', size = 13.5),
          axis.text.x = element_text(angle = x_axis_angle, hjust = x_axis_hjust))

  print(plot)  # Displaying the plot

}



dt_datatable_wrapper <- function(data, n_max =100){

  if(nrow(data)>n_max){
    data <- head(data, n_max)
  }

  dt <- DT::datatable(data, rownames = FALSE,options = list(
    columnDefs = list(
      list(targets = "_all", render = DT::JS(
        "function(data, type, row, meta) {
         if (type === 'display' && typeof data === 'number') {
           if (Math.abs(data) < 1e-3) {
             return data.toExponential(2);  // Display in standard form with 2 decimal places
           } else if (Math.abs(data) < 100) {
             return data.toFixed(2);  // Display with 2 decimal places
           } else {
             return Math.round(data);  // Display as an integer
           }
         }
         return data;
      }")
      )
    )
  ))

  return(dt)
}
