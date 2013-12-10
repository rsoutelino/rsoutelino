# -*- coding: utf-8 -*-
# track changes
from gluon.custom_import import track_changes
track_changes(True)

from gluon.tools import Auth, Service
from helpers.mail import Mailer

service = Service()
private_service = Service()

# DATABASE
db = DAL(**config.db)

# dbsession = DAL("postgres://...")
# session.connect(request, response, dbsession, masterapp='sistema')


# MAILER
mail = Mailer()

# Auth
auth = Auth(db, hmac_key=Auth.get_or_create_key())
auth.settings.extra_fields['auth_user'] = \
    config.auth.settings.extra_fields.auth_user

auth.settings.mailer = mail
auth.settings.registration_requires_verification = \
    config.auth.settings.registration_requires_verification
auth.settings.registration_requires_approval = \
    config.auth.settings.registration_requires_approval
auth.settings.formstyle = config.auth.settings.formstyle

auth.settings.register_onaccept = \
    lambda form: mail.my_mail_sender(template="welcome",
                                     context=form.vars,
                                     to=form.vars.email,
                                     subject="Welcome to Rafael Soutelino \
                                     	site")

auth.define_tables()
