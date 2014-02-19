# -*- coding: utf-8 -*-

from gluon import *
from gluon import current
from gluon.tools import Mail


class Mailer(Mail):
    def __init__(self, config=None):
        Mail.__init__(self)
        self.config = config or current.config
        self.settings.server = self.config.mail.server
        self.settings.sender = self.config.mail.sender
        self.settings.login = self.config.mail.login
        # tratamento de exceptions

    def my_mail_sender(self, template, context, to, subject):
        message = self.my_template_render(template, context)
        try:
            return self.send(to=to,
                             subject=subject,
                             message=message)
        except Exception:
            return "error"  # Gravar um log

    def my_template_render(self, template, context):
        return current.response.render("email_templates/%s.html" % template, context)