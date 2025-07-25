variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ai-training-pipeline"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "account_id" {
  description = "AWS Account ID"
  type        = string
  default     = ""
}

# S3 Configuration
variable "video_processing_bucket" {
  description = "S3 bucket for video processing"
  type        = string
  default     = ""
}

variable "patients_bucket" {
  description = "S3 bucket for patient data and training"
  type        = string
  default     = ""
}

# Lambda Configuration
variable "file_threshold" {
  description = "Number of files required to trigger training"
  type        = number
  default     = 10
}

variable "size_threshold_mb" {
  description = "Size threshold in MB to trigger training"
  type        = number
  default     = 5
}

variable "cooldown_hours" {
  description = "Cooldown period in hours between training runs"
  type        = number
  default     = 1
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 512
}

# Step Functions Configuration
variable "training_timeout_minutes" {
  description = "Step Functions training timeout in minutes"
  type        = number
  default     = 60
}

# SageMaker Configuration
variable "sagemaker_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.m5.large"
}

variable "sagemaker_instance_count" {
  description = "Number of SageMaker instances"
  type        = number
  default     = 1
}

# Video Processing Configuration
variable "video_processing_lambda_timeout" {
  description = "Video processing Lambda timeout in seconds"
  type        = number
  default     = 900
}

variable "video_processing_lambda_memory" {
  description = "Video processing Lambda memory in MB"
  type        = number
  default     = 3008
}

# Notification Configuration
variable "notification_email" {
  description = "Email for notifications"
  type        = string
  default     = ""
}

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs retention"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention in days"
  type        = number
  default     = 30
}