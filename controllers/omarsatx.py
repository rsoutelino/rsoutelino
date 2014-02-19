
def inscricao():
	form = SQLFORM(Formulario).process()
	return dict(form=form)

def download():
	return response.download(request, db, attachment=False)