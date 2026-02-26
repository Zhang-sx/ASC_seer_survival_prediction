# app.R
# 腺鳞癌个体化生存预测 Shiny 应用（动态输入版，带中文标签）
library(shiny)
library(dplyr)
library(xgboost)
library(randomForest)
library(glmnet)
library(gbm)
library(nnet)
install.packages('rsconnect')

# ----------------------------- 加载模型和元数据 -----------------------------
model_dir <- "./models"

# 获取所有元数据文件
metadata_files <- list.files(model_dir, pattern = "_metadata\\.rds$", full.names = TRUE)
if (length(metadata_files) == 0) {
  stop("未找到任何元数据文件，请先运行训练代码生成模型。")
}
metadatas <- list()
for (f in metadata_files) {
  name <- gsub("_metadata\\.rds$", "", basename(f))
  metadatas[[name]] <- readRDS(f)
}

# 获取所有模型文件
model_files <- list.files(model_dir, pattern = "_best\\.rds$", full.names = TRUE)
if (length(model_files) == 0) {
  stop("未找到任何模型文件，请先运行训练代码生成模型。")
}
models <- list()
for (f in model_files) {
  name <- gsub("_best\\.rds$", "", basename(f))
  models[[name]] <- readRDS(f)
}

# 提取所有可用的癌种
available_cancers <- unique(sapply(strsplit(names(metadatas), "_"), `[`, 1))



# 加载模型和元数据后，只保留三个癌种
allowed_cancers <- c("Lung", "Cervix", "Corpus Uteri")
available_cancers <- intersect(available_cancers, allowed_cancers)


# 获取某个癌种的第一个元数据
get_cancer_meta <- function(cancer) {
  key <- grep(paste0("^", cancer, "_"), names(metadatas), value = TRUE)[1]
  if (!is.na(key)) metadatas[[key]] else NULL
}

# 变量名到中文标签的映射
var_labels <- list(
  "Age_factor" = "年龄组",
  "Surgery_group" = "手术",
  "Chemotherapy_factor" = "化疗",
  "Radiation_factor" = "放疗",
  "SEER_stage_factor" = "SEER分期",
  "N_factor" = "N分期",
  "M_factor" = "M分期",
  "T_factor" = "T分期",
  "Liver_M" = "肝转移"
)

# ----------------------------- UI界面 -----------------------------
ui <- fluidPage(
  titlePanel("腺鳞癌个体化生存预测"),
  sidebarLayout(
    sidebarPanel(
      selectInput("cancer", "选择癌种", choices = available_cancers),
      hr(),
      h4("输入患者特征"),
      # 连续变量（所有癌种都可能需要）
      numericInput("Year_dx", "诊断年份", value = 2015, min = 2000, max = 2022),
      numericInput("Nodes_pos", "阳性淋巴结数", value = 0, min = 0, max = 50),
      numericInput("Total_malignant", "恶性肿瘤总数", value = 1, min = 1, max = 10),
      numericInput("Months_to_tx", "治疗等待时间 (月)", value = 0, min = 0, max = 12),
      
      # 动态生成的因子变量输入区域
      uiOutput("dynamic_inputs"),
      
      actionButton("predict", "预测1-5年生存率", class = "btn-primary")
    ),
    mainPanel(
      h4("预测结果"),
      tableOutput("surv_table"),
      verbatimTextOutput("debug")
    )
  )
)

# ----------------------------- 服务器逻辑 -----------------------------
server <- function(input, output, session) {
  
  current_cancer <- reactive({ input$cancer })
  
  current_meta <- reactive({ get_cancer_meta(current_cancer()) })
  
  output$dynamic_inputs <- renderUI({
    meta <- current_meta()
    req(meta)
    
    factor_vars <- names(meta$factor_levels)[!sapply(meta$factor_levels, is.null)]
    
    input_list <- lapply(factor_vars, function(var) {
      label <- ifelse(var %in% names(var_labels), var_labels[[var]], var)
      choices <- meta$factor_levels[[var]]
      selectInput(inputId = var, label = label, choices = choices, selected = choices[1])
    })
    
    do.call(tagList, input_list)
  })
  
  observeEvent(input$predict, {
    cancer <- current_cancer()
    meta <- current_meta()
    req(meta)
    
    # 构建患者数据框
    patient <- data.frame(
      Year_dx = input$Year_dx,
      Nodes_pos = input$Nodes_pos,
      Total_malignant = input$Total_malignant,
      Months_to_tx = input$Months_to_tx,
      stringsAsFactors = FALSE
    )
    
    factor_vars <- names(meta$factor_levels)[!sapply(meta$factor_levels, is.null)]
    for (var in factor_vars) {
      val <- input[[var]]
      if (!is.null(val)) {
        patient[[var]] <- factor(val, levels = meta$factor_levels[[var]])
      }
    }
    
    # 获取该癌种所有时间点
    meta_keys <- grep(paste0("^", cancer, "_"), names(metadatas), value = TRUE)
    time_strings <- gsub(paste0(cancer, "_"), "", meta_keys)
    time_points <- sort(as.numeric(time_strings))
    
    surv_probs <- rep(NA, length(time_points))
    names(surv_probs) <- paste0(time_points, "个月")
    
    for (i in seq_along(time_points)) {
      tp <- time_points[i]
      meta_key <- paste0(cancer, "_", tp)
      meta_tp <- metadatas[[meta_key]]
      if (is.null(meta_tp)) next
      
      model_key <- paste0(cancer, "_", tp, "months")
      model <- models[[model_key]]
      if (is.null(model)) next
      
      vars_needed <- meta_tp$variables
      patient_sub <- patient[, intersect(vars_needed, names(patient)), drop = FALSE]
      
      # 确保因子水平正确
      for (v in names(patient_sub)) {
        if (is.factor(patient_sub[[v]]) && !is.null(meta_tp$factor_levels[[v]])) {
          patient_sub[[v]] <- factor(patient_sub[[v]], levels = meta_tp$factor_levels[[v]])
        }
      }
      
      model_type <- meta_tp$model_type
      pred_death <- NULL
      
      tryCatch({
        if (model_type == "LR") {
          pred_death <- predict(model, newdata = patient_sub, type = "response")
        } else if (model_type == "XGB") {
          patient_mat <- patient_sub
          for (v in names(patient_mat)) {
            if (is.factor(patient_mat[[v]])) patient_mat[[v]] <- as.numeric(patient_mat[[v]])
          }
          dmat <- xgb.DMatrix(data = as.matrix(patient_mat))
          pred_death <- predict(model, dmat)
        } else if (model_type == "LASSO") {
          x_patient <- model.matrix(~ . - 1, data = patient_sub)
          pred_death <- as.numeric(predict(model, newx = x_patient, type = "response"))
        } else if (model_type == "SGBT") {
          patient_gbm <- patient_sub
          for (v in names(patient_gbm)) {
            if (is.factor(patient_gbm[[v]])) patient_gbm[[v]] <- as.numeric(patient_gbm[[v]])
          }
          pred_death <- predict(model, newdata = patient_gbm, n.trees = 100, type = "response")
        } else if (model_type == "NNET") {
          patient_nnet <- patient_sub
          for (v in names(patient_nnet)) {
            if (is.factor(patient_nnet[[v]])) patient_nnet[[v]] <- as.numeric(patient_nnet[[v]])
          }
          pred_death <- predict(model, newdata = patient_nnet)
        } else if (model_type == "RF") {
          patient_rf <- patient_sub
          for (v in names(patient_rf)) {
            if (is.factor(patient_rf[[v]])) patient_rf[[v]] <- as.numeric(patient_rf[[v]])
          }
          pred_death <- predict(model, newdata = patient_rf, type = "prob")[, "1"]
        } else if (model_type == "KNN") {
          patient_knn <- patient_sub
          for (v in names(patient_knn)) {
            if (is.factor(patient_knn[[v]])) patient_knn[[v]] <- as.numeric(patient_knn[[v]])
          }
          pred_death <- predict(model, newdata = patient_knn, type = "prob")[, "1"]
        }
      }, error = function(e) {})
      
      if (!is.null(pred_death) && length(pred_death) == 1) {
        surv_probs[i] <- 1 - pred_death
      }
    }
    
    if (any(!is.na(surv_probs))) {
      result_df <- data.frame(
        时间点 = names(surv_probs),
        预测生存概率 = round(surv_probs, 3)
      )
      output$surv_table <- renderTable(result_df)
    } else {
      output$surv_table <- renderTable(data.frame(提示 = "无法预测，请检查输入"))
    }
    
    output$debug <- renderPrint({
      cat("患者数据:\n")
      print(patient)
      cat("\n当前癌种元数据变量:\n")
      print(meta$variables)
    })
  })
}

shinyApp(ui, server)

rsconnect::setAccountInfo(name='asc-survival-prediction',
                          token='6890F08E68A1F34CFEE73892B9F8E14E',
                          secret='TnQQkzsFGvcdGLZmoh2gZaOnGZNZaMn79PgcVjNU')

library(rsconnect)
rsconnect::deployApp("C:/Users/zhang/Desktop/Prediction_Models/app.R")