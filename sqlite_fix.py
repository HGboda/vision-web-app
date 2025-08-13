# -*- coding: utf-8 -*-
# This script overrides the default sqlite3 with pysqlite3 to meet ChromaDB requirements
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
