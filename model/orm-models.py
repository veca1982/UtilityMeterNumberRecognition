import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime, Sequence
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
import datetime


Base = declarative_base()

class Korisnik(Base):
    __tablename__ = 'korisnici'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(250), nullable=False)
    oib = Column(String(11))
    device_ip_adress = Column(String(100))
    salary = Column(Float)
    tstamp = Column(DateTime)

    def __init__(self, name, oib, device_ip_adress, salary, tstapm):
        self.name = name
        self.oib = oib
        self.device_ip_adress = device_ip_adress
        self.salary = salary
        self.tstamp = tstapm

    def __repr__(self):
        return '<id {}>'.format(self.id)

class LastPicTaken(Base):
    __tablename__ = 'last_pic_taken_for_korisnik'
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, Sequence('last_pic_taken_id_seq'), primary_key=True)
    id_korisnika = Column(Integer, ForeignKey('korisnici.id'))
    status = Column(String)
    time_last_pic_taken = Column(DateTime)
    user = relationship("Korisnik", back_populates="last_pics_taken")

    def __init__(self, status, time_last_pic_taken):
        self.status = status
        self.time_last_pic_taken = time_last_pic_taken

    def __repr__(self):
        return '<id {}>'.format(self.id)

Korisnik.last_pics_taken = relationship("LastPicTaken", order_by=LastPicTaken.id, back_populates="user")

# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine('postgresql://postgres:ivavedran@localhost:5432/utilitymeter')

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)


# SQLAlchemy SESSIONS

'''Sessions give you access to Transactions, whereby on success you can commit the transaction
or rollback one incase you encounter an error'''

Session = sessionmaker(bind=engine)
session = Session()

# Insert multiple data in this session, similarly you can delete
user1 = Korisnik('Marko', '54941759440', '192.168.1.66', 3200.43, datetime.datetime.utcnow())
user2 = Korisnik('Ivan', '54941759441', '192.168.1.67', 3500.43, datetime.datetime.utcnow())

user1.last_pics_taken = [LastPicTaken(status='0', time_last_pic_taken=datetime.datetime.utcnow())]

session.add(user1)
session.add(user2)

try:
    #print 'bla'
    session.commit()
# You can catch exceptions with &nbsp;SQLAlchemyError base class
except SQLAlchemyError as e:
    session.rollback()
    print (str(e))

#Get data
for user in session.query(Korisnik).all():
    print ("name of the user is", user.name)
    print ("oib of the user is", user.oib)

#rolback for test
session.rollback()
#Close the connection
engine.dispose()