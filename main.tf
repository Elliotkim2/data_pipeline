# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local values
locals {
  account_id = var.account_id != "" ? var.account_id : data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  
  # Generate bucket names if not provided
  video_processing_bucket = var.video_processing_bucket != "" ? var.video_processing_bucket : "${var.project_name}-video-processing-${local.account_id}"
  patients_bucket        = var.patients_bucket != "" ? var.patients_bucket : "${var.project_name}-patients-${local.account_id}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# IAM Module
module "iam" {
  source = "./modules/iam"
  
  project_name            = var.project_name
  environment             = var.environment
  account_id              = local.account_id
  region                  = local.region
  video_processing_bucket = local.video_processing_bucket
  patients_bucket         = local.patients_bucket
  
  tags = local.common_tags
}

# S3 Module
module "s3" {
  source = "./modules/s3"
  
  project_name            = var.project_name
  environment             = var.environment
  video_processing_bucket = local.video_processing_bucket
  patients_bucket         = local.patients_bucket
  
  # Lambda ARNs for S3 notifications
  video_processor_lambda_arn = module.lambda.video_processor_lambda_arn
  process_upload_lambda_arn  = module.lambda.process_upload_lambda_arn
  
  tags = local.common_tags
}

# Lambda Module
module "lambda" {
  source = "./modules/lambda"
  
  project_name            = var.project_name
  environment             = var.environment
  video_processing_bucket = local.video_processing_bucket
  patients_bucket         = local.patients_bucket
  training_bucket         = local.training_bucket
  
  # Lambda configuration
  file_threshold         = var.file_threshold
  size_threshold_mb      = var.size_threshold_mb
  cooldown_hours         = var.cooldown_hours
  lambda_timeout         = var.lambda_timeout
  lambda_memory          = var.lambda_memory
  video_lambda_timeout   = var.video_processing_lambda_timeout
  video_lambda_memory    = var.video_processing_lambda_memory
  
  # IAM roles
  video_processing_role_arn = module.iam.video_processing_role_arn
  ai_training_lambda_role_arn = module.iam.ai_training_lambda_role_arn
  
  # Step Functions ARN
  state_machine_arn = module.stepfunctions.state_machine_arn
  
  # CloudWatch logs
  enable_cloudwatch_logs = var.enable_cloudwatch_logs
  log_retention_days     = var.log_retention_days
  
  tags = local.common_tags
}

# Step Functions Module
module "stepfunctions" {
  source = "./modules/stepfunctions"
  
  project_name    = var.project_name
  environment     = var.environment
  patients_bucket = local.patients_bucket
  
  # Lambda ARNs
  training_lambda_arn = module.lambda.training_lambda_arn
  
  # SageMaker configuration
  sagemaker_instance_type  = var.sagemaker_instance_type
  sagemaker_instance_count = var.sagemaker_instance_count
  
  # IAM role
  stepfunctions_role_arn = module.iam.stepfunctions_role_arn
  
  # Timeout
  training_timeout_minutes = var.training_timeout_minutes
  
  tags = local.common_tags
}