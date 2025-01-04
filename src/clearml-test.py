from clearml import Task
task = Task.init(project_name="Test Project", task_name="Test Task")
task.close()