# IAM Role for Video Processing Lambda
resource "aws_iam_role" "video_processing_role" {
  name = "${var.project_name}-video-processing-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Role for AI Training Lambda Functions
resource "aws_iam_role" "ai_training_lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Role for Step Functions
resource "aws_iam_role" "stepfunctions_role" {
  name = "${var.project_name}-stepfunctions-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Video Processing Lambda Policy
resource "aws_iam_policy" "video_processing_policy" {
  name        = "${var.project_name}-video-processing-policy-${var.environment}"
  description = "Policy for video processing Lambda function"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.video_processing_bucket}/*",
          "arn:aws:s3:::${var.patients_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.video_processing_bucket}",
          "arn:aws:s3:::${var.patients_bucket}"
        ]
      }
    ]
  })

  tags = var.tags
}

# AI Training Lambda Policy
resource "aws_iam_policy" "ai_training_lambda_policy" {
  name        = "${var.project_name}-ai-training-lambda-policy-${var.environment}"
  description = "Policy for AI training Lambda functions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.patients_bucket}",
          "arn:aws:s3:::${var.patients_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query"
        ]
        Resource = [
          "arn:aws:dynamodb:${var.region}:${var.account_id}:table/emotion_sates",
          "arn:aws:dynamodb:${var.region}:${var.account_id}:table/emotion_sates/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution"
        ]
        Resource = "arn:aws:states:${var.region}:${var.account_id}:stateMachine:${var.project_name}-training-pipeline-${var.environment}"
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = "arn:aws:bedrock:${var.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
      }
    ]
  })

  tags = var.tags
}

# AI Training Lambda SageMaker Policy
resource "aws_iam_policy" "ai_training_sagemaker_policy" {
  name        = "${var.project_name}-ai-training-sagemaker-policy-${var.environment}"
  description = "SageMaker policy for AI training Lambda functions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:ListTrainingJobs"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_execution_role.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

# Step Functions Policy
resource "aws_iam_policy" "stepfunctions_policy" {
  name        = "${var.project_name}-stepfunctions-policy-${var.environment}"
  description = "Policy for Step Functions state machine"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          "arn:aws:lambda:${var.region}:${var.account_id}:function:${var.project_name}-*-${var.environment}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account_id}:*"
      }
    ]
  })

  tags = var.tags
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "video_processing_basic_execution" {
  role       = aws_iam_role.video_processing_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "video_processing_transcribe_access" {
  role       = aws_iam_role.video_processing_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonTranscribeFullAccess"
}

resource "aws_iam_role_policy_attachment" "stepfunctions_full_access" {
  role       = aws_iam_role.stepfunctions_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSStepFunctionsFullAccess"
}

# Attach policies to roles
resource "aws_iam_role_policy_attachment" "video_processing_policy_attachment" {
  role       = aws_iam_role.video_processing_role.name
  policy_arn = aws_iam_policy.video_processing_policy.arn
}

resource "aws_iam_role_policy_attachment" "ai_training_lambda_policy_attachment" {
  role       = aws_iam_role.ai_training_lambda_role.name
  policy_arn = aws_iam_policy.ai_training_lambda_policy.arn
}

resource "aws_iam_role_policy_attachment" "ai_training_sagemaker_policy_attachment" {
  role       = aws_iam_role.ai_training_lambda_role.name
  policy_arn = aws_iam_policy.ai_training_sagemaker_policy.arn
}

resource "aws_iam_role_policy_attachment" "stepfunctions_policy_attachment" {
  role       = aws_iam_role.stepfunctions_role.name
  policy_arn = aws_iam_policy.stepfunctions_policy.arn
}

# SageMaker execution role for training jobs
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# SageMaker execution policy
resource "aws_iam_policy" "sagemaker_execution_policy" {
  name        = "${var.project_name}-sagemaker-execution-policy-${var.environment}"
  description = "Policy for SageMaker execution role"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.patients_bucket}",
          "arn:aws:s3:::${var.patients_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy_attachment" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_execution_policy.arn
}