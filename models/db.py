# -*- coding: utf-8 -*-

# db = DAL("sqlite://forms.sqlite", check_reserved=["all"])

Contacts = db.define_table("contacts",
	Field("name", notnull=True),
	Field("email", requires=IS_EMAIL()),
	Field("tel"),
	Field("adress"),
	Field("commentary", "text") )

Pictures = db.define_table("pictures",
	Field("thumb", "upload"),
	Field("foto", "upload"))

Comment = db.define_table("coment",
	Field("name", requires=IS_NOT_EMPTY()),
	Field("coment", "text", required=True))

UserAddress = db.define_table("user_address",
	Field("user_id"),
	Field("zip_code"),
	Field("country"),
	Field("region"),
	Field("provincy"),
	Field("city"),
	Field("address_type"),
	Field("address"),
	Field("home_number"),
	Field("extra"),
	format="%(region)s")


Arquivo = db.define_table("arquivo",
    Field("file_arquivo", "upload"))

Formulario = db.define_table("omarsat",
    Field("nome"),
	Field("instituicao"),
	Field("rua"),
	Field("bairro"),
	Field("cidade"),
	Field("estado"),
	Field("cep"),
	Field("telefone"),
	Field("fax"),
	Field("email", requires=IS_EMAIL()),
	Field("apresentacao","boolean"),
	Field("titulo_trab"),
	Field("estudante","boolean"),
	Field("resumo","upload"))



class NOT_STARTS_WITH(object):
	def __init__(self, error_message="nao permitido iniciar com %s", letter="a"):
		self.error_message = error_message
		self.letter = letter

	def __call__(self, value):
		# (value, None) # passou
		# (value, error_message) # não passou
		if value.startswith(self.letter):
			return (value, self.error_message % self.letter)
		else:
			return (value, None)


## Category
#Contacts.name.requires = [IS_NOT_EMPTY(error_message="vc deve preencher"),
#                          IS_NOT_IN_DB(db, 'Contacts.name')]
# Contacts.description.represent = lambda value, row: XML(value)
Contacts.name.label = "Contacts"
Contacts.name.comment = "Your name"

#auth user
db.auth_user.avatar.label = "sua foto"

# Order
dbset = db(UserAddress.user_id == auth.user_id)
#Contacts.name.requires = IS_IN_DB(dbset, "user_address.id", "%(country)s - %(city)s")