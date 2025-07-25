output "video_processor_lambda_arn" {
  description = "ARN of the video processor Lambda function"
  value       = aws_lambda_function.video_processor.arn
}

output "process_upload_lambda_arn" {
  description = "ARN of the process upload Lambda function"
  value       = aws_lambda_function.process_upload.arn
}

output "training_agent_lambda_arn" {
  description = "ARN of the training agent Lambda function"
  value       = aws_lambda_function.training_agent.arn
}

output "video_processor_lambda_name" {
  description = "Name of the video processor Lambda function"
  value       = aws_lambda_function.video_processor.function_name
}

output "process_upload_lambda_name" {
  description = "Name of the process upload Lambda function"
  value       = aws_lambda_function.process_upload.function_name
}

output "training_agent_lambda_name" {
  description = "Name of the training agent Lambda function"
  value       = aws_lambda_function.training_agent.function_name
}
