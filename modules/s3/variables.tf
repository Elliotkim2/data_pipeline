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

variable "video_processor_lambda_arn" {
  description = "ARN of the video processor Lambda function"
  type        = string
}

variable "process_upload_lambda_arn" {
  description = "ARN of the process upload Lambda function"
  type        = string
}

variable "video_processor_lambda_name" {
  description = "Name of the video processor Lambda function"
  type        = string
}

variable "process_upload_lambda_name" {
  description = "Name of the process upload Lambda function"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}