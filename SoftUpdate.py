def soft_update(teacher_model, student_model, tau):
    """
    Soft update model parameters.
    θ_teacher = τ*θ_student + (1 - τ)*θ_teacher

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    :param tau: interpolation parameter (0.001 in your case)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(tau*student_param.data + (1.0-tau)*teacher_param.data)