# Lambda function for video processing
resource "aws_lambda_function" "video_processor" {
  filename         = data.archive_file.video_processor_zip.output_path
  function_name    = "${var.project_name}-video-processor-${var.environment}"
  role            = var.video_processing_role_arn
  handler         = "handler.handler"
  runtime         = "python3.9"
  timeout         = var.video_lambda_timeout
  memory_size     = var.video_lambda_memory
  source_code_hash = data.archive_file.video_processor_zip.output_base64sha256

  environment {
    variables = {
      PATIENTS_BUCKET = var.patients_bucket
      VIDEO_BUCKET    = var.video_processing_bucket
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.video_processor_logs,
  ]

  tags = var.tags
}

# Lambda function for processing uploads
resource "aws_lambda_function" "process_upload" {
  filename         = data.archive_file.process_upload_zip.output_path
  function_name    = "${var.project_name}-process-upload-${var.environment}"
  role            = var.ai_training_lambda_role_arn
  handler         = "handler.handler"
  runtime         = "python3.9"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory
  source_code_hash = data.archive_file.process_upload_zip.output_base64sha256

  environment {
    variables = {
      STATE_MACHINE_ARN = var.state_machine_arn
      FILE_THRESHOLD    = var.file_threshold
      SIZE_THRESHOLD_MB = var.size_threshold_mb
      COOLDOWN_HOURS    = var.cooldown_hours
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.process_upload_logs,
  ]

  tags = var.tags
}

# Lambda function for training agent
resource "aws_lambda_function" "training_agent" {
  filename         = data.archive_file.training_agent_zip.output_path
  function_name    = "${var.project_name}-training-agent-${var.environment}"
  role            = var.ai_training_lambda_role_arn
  handler         = "handler.handler"
  runtime         = "python3.9"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory
  source_code_hash = data.archive_file.training_agent_zip.output_base64sha256

  environment {
    variables = {
      PATIENTS_BUCKET = var.patients_bucket
      SAGEMAKER_ROLE  = var.sagemaker_execution_role_arn
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.training_agent_logs,
  ]

  tags = var.tags
}

# Archive files for Lambda deployment
data "archive_file" "video_processor_zip" {
  type        = "zip"
  source_dir  = "${path.root}/lambda-code/video-processor"
  output_path = "${path.root}/lambda-code/video-processor.zip"
}

data "archive_file" "process_upload_zip" {
  type        = "zip"
  source_dir  = "${path.root}/lambda-code/process-upload"
  output_path = "${path.root}/lambda-code/process-upload.zip"
}

data "archive_file" "training_agent_zip" {
  type        = "zip"
  source_dir  = "${path.root}/lambda-code/training-agent"
  output_path = "${path.root}/lambda-code/training-agent.zip"
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "video_processor_logs" {
  count             = var.enable_cloudwatch_logs ? 1 : 0
  name              = "/aws/lambda/${var.project_name}-video-processor-${var.environment}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "process_upload_logs" {
  count             = var.enable_cloudwatch_logs ? 1 : 0
  name              = "/aws/lambda/${var.project_name}-process-upload-${var.environment}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "training_agent_logs" {
  count             = var.enable_cloudwatch_logs ? 1 : 0
  name              = "/aws/lambda/${var.project_name}-training-agent-${var.environment}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}