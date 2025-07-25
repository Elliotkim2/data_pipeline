variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "patients_bucket" {
  description = "S3 bucket for patient data"
  type        = string
}

variable "training_lambda_arn" {
  description = "ARN of the training Lambda function"
  type        = string
}

variable "stepfunctions_role_arn" {
  description = "ARN of the Step Functions role"
  type        = string
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