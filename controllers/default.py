
def index():
	return dict()

def user():
	from gluon.storage import Storage
	objects = Storage()
	if request.args(0) in ['register', 'login']:
		objects.login = auth.login()
		objects.register = auth.register()
	else:
		objects.form = auth()
	return dict(objects=objects)

def formpython():
    tmp = db(Arquivo).select()
    return locals()

def students():
    return dict()

def datapanel():
    return dict()

def download():
    return response.download(request, db, attachment=False)

def arquivos():
    pass

def inscricao():
    form = SQLFORM.grid(db.omarsat).process()
    return dict(form=form)

def omarsat():
    form = SQLFORM(Formulario).process()
    return dict(form=form)
