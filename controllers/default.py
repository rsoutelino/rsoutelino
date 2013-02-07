
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
    return locals()