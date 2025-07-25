# Video Processing S3 Bucket
resource "aws_s3_bucket" "video_processing" {
  bucket = var.video_processing_bucket
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "video_processing" {
  bucket = aws_s3_bucket.video_processing.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "video_processing" {
  bucket = aws_s3_bucket.video_processing.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "video_processing" {
  bucket = aws_s3_bucket.video_processing.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Patients S3 Bucket
resource "aws_s3_bucket" "patients" {
  bucket = var.patients_bucket
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "patients" {
  bucket = aws_s3_bucket.patients.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "patients" {
  bucket = aws_s3_bucket.patients.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "patients" {
  bucket = aws_s3_bucket.patients.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Notification for Video Processing
resource "aws_s3_bucket_notification" "video_processing_notification" {
  bucket = aws_s3_bucket.video_processing.id

  lambda_function {
    lambda_function_arn = var.video_processor_lambda_arn
    events             = ["s3:ObjectCreated:*"]
    filter_suffix      = ".MOV"
  }

  lambda_function {
    lambda_function_arn = var.video_processor_lambda_arn
    events             = ["s3:ObjectCreated:*"]
    filter_suffix      = ".mp4"
  }

  lambda_function {
    lambda_function_arn = var.video_processor_lambda_arn
    events             = ["s3:ObjectCreated:*"]
    filter_suffix      = ".avi"
  }

  depends_on = [aws_lambda_permission.s3_invoke_video_processor]
}

# S3 Bucket Notification for Patients (Training Data)
resource "aws_s3_bucket_notification" "patients_notification" {
  bucket = aws_s3_bucket.patients.id

  lambda_function {
    lambda_function_arn = var.process_upload_lambda_arn
    events             = ["s3:ObjectCreated:*"]
    filter_suffix      = ".json"
  }

  depends_on = [aws_lambda_permission.s3_invoke_process_upload]
}

# Lambda permission for S3 to invoke video processor
resource "aws_lambda_permission" "s3_invoke_video_processor" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = var.video_processor_lambda_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.video_processing.arn
}

# Lambda permission for S3 to invoke process upload
resource "aws_lambda_permission" "s3_invoke_process_upload" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = var.process_upload_lambda_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.patients.arn
}