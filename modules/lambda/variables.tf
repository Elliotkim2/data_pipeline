variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "video_processing_bucket" {
  description = "S3 bucket for video processing"
  type        = string
}

variable "patients_bucket" {
  description = "S3 bucket for patient data"
  type        = string
}

variable "file_threshold" {
  description = "Number of files required to trigger training"
  type        = number
}

variable "size_threshold_mb" {
  description = "Size threshold in MB to trigger training"
  type        = number
}

variable "cooldown_hours" {
  description = "Cooldown period in hours between training runs"
  type        = number
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
}

variable "video_lambda_timeout" {
  description = "Video processing Lambda timeout in seconds"
  type        = number
}

variable "video_lambda_memory" {
  description = "Video processing Lambda memory in MB"
  type        = number
}

variable "video_processing_role_arn" {
  description = "ARN of the video processing Lambda role"
  type        = string
}

variable "ai_training_lambda_role_arn" {
  description = "ARN of the AI training Lambda role"
  type        = string
}

variable "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  type        = string
}

variable "state_machine_arn" {
  description = "ARN of the Step Functions state machine"
  type        = string
}

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs retention"
  type        = bool
}

variable "log_retention_days" {
  description = "CloudWatch logs retention in days"
  type        = number
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}