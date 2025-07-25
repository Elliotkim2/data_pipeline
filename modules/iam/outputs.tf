output "video_processing_role_arn" {
  description = "ARN of the video processing Lambda role"
  value       = aws_iam_role.video_processing_role.arn
}

output "ai_training_lambda_role_arn" {
  description = "ARN of the AI training Lambda role"
  value       = aws_iam_role.ai_training_lambda_role.arn
}

output "stepfunctions_role_arn" {
  description = "ARN of the Step Functions role"
  value       = aws_iam_role.stepfunctions_role.arn
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "video_processing_role_name" {
  description = "Name of the video processing Lambda role"
  value       = aws_iam_role.video_processing_role.name
}

output "ai_training_lambda_role_name" {
  description = "Name of the AI training Lambda role"
  value       = aws_iam_role.ai_training_lambda_role.name
}

output "stepfunctions_role_name" {
  description = "Name of the Step Functions role"
  value       = aws_iam_role.stepfunctions_role.name
}

output "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.name
}