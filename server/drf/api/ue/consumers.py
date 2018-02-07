"""
import json
from channels import Group
from channels.auth import channel_session_user, channel_session_user_from_http
import threading
import random

def sendmsg(num):
    Group('stocks').send({'text':num})

t = 0

def periodic():
    global t;
    n = random.randint(100,200);
    sendmsg(str(n))
    t = threading.Timer(1, periodic)
    t.start()

def ws_message(message):
    global t
    # ASGI WebSocket packet-received and send-packet message types
    # both have a "text" key for their textual data.
    print(message.content['text'])
    if ( message.content['text'] == "start"):
        periodic()
    else:
        t.cancel()
   # message.reply_channel.send({'text':'200'})

def ws_connect(message):
    Group('stocks').add(message.reply_channel)
    Group('stocks').send({'text':'connected'})


def ws_disconnect(message):
    Group('stocks').send({'text':'disconnected'})
    Group('stocks').discard(message.reply_channel)
"""

"""
class ChatConsumer(WebsocketConsumer):

    def connect(self):
        # Make a database row with our channel name
        Clients.objects.create(channel_name=self.channel_name)

    def disconnect(self):
        # Note that in some rare cases (power loss, etc) disconnect may fail
        # to run; this naive example would leave zombie channel names around.
        Clients.objects.filter(channel_name=self.channel_name).delete()

    def chat_message(self, event):
        # Handles the "chat.message" event when it's sent to us.
        self.send(text_data=event["text"])
"""

from channels.generic.websocket import JsonWebsocketConsumer
from channels.layers import get_channel_layer

class MeasurementConsumer(JsonWebsocketConsumer):

    def connect(self):
        # Called on connection. Either call
        self.accept()
        self.send_json(content="Hello world!")

    def receive_json(self, content=None):
        self.send_json(content=content)

    def disconnect(self, close_code):
        # Called when the socket closes
        print('done')
