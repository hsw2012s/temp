# -*- coding: utf-8 -*- 
import psycopg2 as pg2 
conn = None 
try: 
    with pg2.connect("host = localhost dbname=ATG user=postgres password=1234") as conn: 
        with conn.cursor() as cur: 
            sql = "create table wave(Id integer primary key, Name varchar(20), wavelist int)"  #table제작 구문
            cur.execute(sql) #제작구문 실행(excute)
            cur.execute("INSERT INTO wave VALUES(1,'wulsung3',52642)") #데이터 넣는구문
            cur.execute("INSERT INTO wave VALUES(2,'hanbit',57127)") #데이터 넣는구문
            cur.execute("SELECT * FROM wave") #보여주는 구문
            rows = cur.fetchall() #한번에 모두 실행
        cur.close() 
    conn.commit() #반영
except Exception as e: 
    print ('Error : ', e)
else: 
    print (rows )
finally: 
    if conn: 
        conn.close()
