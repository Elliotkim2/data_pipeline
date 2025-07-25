output "video_processing_bucket" {
  description = "Name of the video processing S3 bucket"
  value       = module.s3.video_processing_bucket
}

output "patients_bucket" {
  description = "Name of the patients S3 bucket (also used for training)"
  value       = module.s3.patients_bucket
}

output "video_processor_lambda_arn" {
  description = "ARN of the video processor Lambda function"
  value       = module.lambda.video_processor_lambda_arn
}

output "process_upload_lambda_arn" {
  description = "ARN of the process upload Lambda function"
  value       = module.lambda.process_upload_lambda_arn
}

output "training_lambda_arn" {
  description = "ARN of the training Lambda function"
  value       = module.lambda.training_lambda_arn
}

output "state_machine_arn" {
  description = "ARN of the Step Functions state machine"
  value       = module.stepfunctions.state_machine_arn
}

output "lambda_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = module.iam.lambda_role_arn
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    region           = local.region
    account_id       = local.account_id
    project_name     = var.project_name
    environment      = var.environment
    file_threshold   = var.file_threshold
    cooldown_hours   = var.cooldown_hours
  }
}