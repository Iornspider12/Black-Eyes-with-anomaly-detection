import smtplib

server= smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login('rtechleap@gmail.com','IornspideR')
server.sendmail('rtechleap@gmail.com','iornspider121212@gmail.com','mail sent for verification')
print('sent')