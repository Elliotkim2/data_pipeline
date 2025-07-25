# Step Functions State Machine
resource "aws_sfn_state_machine" "training_pipeline" {
  name     = "${var.project_name}-training-pipeline-${var.environment}"
  role_arn = var.stepfunctions_role_arn

  definition = jsonencode({
    Comment = "AI Training Pipeline"
    StartAt = "AnalyzeData"
    States = {
      AnalyzeData = {
        Type     = "Task"
        Resource = var.training_lambda_arn
        Next     = "TrainingDecision"
      }
      TrainingDecision = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.body.decision"
            StringEquals  = "TRAIN"
            Next         = "LogTraining"
          }
        ]
        Default = "SkipTraining"
      }
      LogTraining = {
        Type   = "Pass"
        Result = "Training would start here"
        Next   = "UpdateStatus"
      }
      SkipTraining = {
        Type   = "Pass"
        Result = "Training skipped"
        End    = true
      }
      UpdateStatus = {
        Type   = "Pass"
        Result = "Status updated"
        End    = true
      }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.stepfunctions_logs.arn}:*"
    include_execution_data = true
    level                  = "ERROR"
  }

  tags = var.tags
}

# CloudWatch Log Group for Step Functions
resource "aws_cloudwatch_log_group" "stepfunctions_logs" {
  name              = "/aws/stepfunctions/${var.project_name}-training-pipeline-${var.environment}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}