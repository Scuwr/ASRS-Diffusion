start-redis:
	redis-7.0.8/src/redis-server redis.conf

install:
	wget https://download.redis.io/releases/redis-7.0.8.tar.gz
	tar xvzf redis-7.0.8.tar.gz
	rm redis-7.0.8.tar.gz
	(cd redis-7.0.8; make)