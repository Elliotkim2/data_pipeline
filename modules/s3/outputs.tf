output "video_processing_bucket" {
  description = "Name of the video processing S3 bucket"
  value       = aws_s3_bucket.video_processing.bucket
}

output "patients_bucket" {
  description = "Name of the patients S3 bucket"
  value       = aws_s3_bucket.patients.bucket
}

output "video_processing_bucket_arn" {
  description = "ARN of the video processing S3 bucket"
  value       = aws_s3_bucket.video_processing.arn
}

output "patients_bucket_arn" {
  description = "ARN of the patients S3 bucket"
  value       = aws_s3_bucket.patients.arn
}