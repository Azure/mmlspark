// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.hadoop

import org.apache.hadoop.conf.Configuration
import org.slf4j.Logger
import scala.language.existentials
import scala.sys.process._

class HadoopUtils(log: Logger, hadoopConf: Configuration) {
  // Is there a better way? We need to deduce full Hadoop conf
  // including current active namenode etc as well as YARN properties
  // going forward anyway for cluster participation in GPU-YARN-queue mode
  // fs.defaultFS isn't good on HDI because we rewrite to WASB
  // Answer, Slightly better:
  /*
    $ hdfs getconf -confKey dfs.nameservices
    mycluster
    $ hdfs getconf -confKey dfs.ha.namenodes.mycluster
    nn1,nn2
    $ hdfs haadmin -getServiceState nn1
    active
    $ hdfs haadmin -getServiceState nn2
    standby
  */
  private val NAMESERVICES_KEY = "dfs.nameservices"
  private val NAMENODE_KEY_ROOT = "dfs.ha.namenodes"
  private val RPC_KEY_ROOT = "dfs.namenode.rpc-address"

  private def getNameServices: String = {
    hadoopConf.get(NAMESERVICES_KEY)
  }

  private def getNameNodes: Seq[String] = {
    val nameservices = getNameServices
    log.info(s"Nameservices for cluster at '$nameservices'")
    hadoopConf.get(combine(NAMENODE_KEY_ROOT, nameservices)).split(",")
  }

  private def isActiveNode(namenode: String): Boolean = {
    shellout(s"hdfs haadmin -getServiceState $namenode").startsWith("active")
  }

  private def combine(keys: String*): String = keys.mkString(".")

  def getActiveNameNode: String = {
    val nameservices = getNameServices
    log.info(s"Nameservices for cluster at '$nameservices'")
    val namenodes = getNameNodes
    log.info(s"Querying namenodes:\n${namenodes.foreach(log.info)}")
    val active = namenodes.par
      .filter(isActiveNode)
      .head
    log.info(s"Found $active as active namenode")
    hadoopConf.get(combine(RPC_KEY_ROOT, nameservices, active))
  }

  // This is only to make sure all uses go away ASAP into Process utils
  // I realize this means it will be around forever
  private def shellout(cmd: String): String = {
    log.info(s"Executing external process $cmd...")
    val ret = cmd.!!
    log.info(s"$ret...done!")
    ret
  }

}
