def contact():
    form = SQLFORM(db.contacts,
        labels = {"name":"Nome",
                    "email":"E-mail", "tel":"Telefone",
                    "adress":"Endereco", "commentary":"Comentario"})
    if form.process().accepted:
        message = ("Nome: %(name)s\n"
                    "E-mail: %(email)s\n"
                    "Telefone: %(tel)s\n"
                    "Endereco: %(adress)s\n"
                    "Comentario: %(commentary)s\n")

        if mail.my_mail_sender(template="welcome", context= form.vars, to="rsoutelino@gmail.com", subject="You received a message at rsoutelino.com"):
            response.flash=T("Your message was successfully sent!")
        else:
            response.flash=T("Ops, something went wrong, try again please...")
    else:
        response.flash="try again"
	return dict(form=form)