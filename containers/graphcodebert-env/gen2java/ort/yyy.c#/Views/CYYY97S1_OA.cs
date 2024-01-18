// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY97S1_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:32
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF EXPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: CYYY97S1_OA
  /// </summary>
  [Serializable]
  public class CYYY97S1_OA : ViewBase, IExportView
  {
    private static CYYY97S1_OA[] freeArray = new CYYY97S1_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP_REFERENCE
    //        Type: IYY1_SERVER_DATA
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpReferenceIyy1ServerDataServerDate
    /// </summary>
    private char _ExpReferenceIyy1ServerDataServerDate_AS;
    /// <summary>
    /// Attribute missing flag for: ExpReferenceIyy1ServerDataServerDate
    /// </summary>
    public char ExpReferenceIyy1ServerDataServerDate_AS {
      get {
        return(_ExpReferenceIyy1ServerDataServerDate_AS);
      }
      set {
        _ExpReferenceIyy1ServerDataServerDate_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpReferenceIyy1ServerDataServerDate
    /// Domain: Date
    /// Length: 8
    /// </summary>
    private int _ExpReferenceIyy1ServerDataServerDate;
    /// <summary>
    /// Attribute for: ExpReferenceIyy1ServerDataServerDate
    /// </summary>
    public int ExpReferenceIyy1ServerDataServerDate {
      get {
        return(_ExpReferenceIyy1ServerDataServerDate);
      }
      set {
        _ExpReferenceIyy1ServerDataServerDate = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpReferenceIyy1ServerDataServerTime
    /// </summary>
    private char _ExpReferenceIyy1ServerDataServerTime_AS;
    /// <summary>
    /// Attribute missing flag for: ExpReferenceIyy1ServerDataServerTime
    /// </summary>
    public char ExpReferenceIyy1ServerDataServerTime_AS {
      get {
        return(_ExpReferenceIyy1ServerDataServerTime_AS);
      }
      set {
        _ExpReferenceIyy1ServerDataServerTime_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpReferenceIyy1ServerDataServerTime
    /// Domain: Time
    /// Length: 6
    /// </summary>
    private int _ExpReferenceIyy1ServerDataServerTime;
    /// <summary>
    /// Attribute for: ExpReferenceIyy1ServerDataServerTime
    /// </summary>
    public int ExpReferenceIyy1ServerDataServerTime {
      get {
        return(_ExpReferenceIyy1ServerDataServerTime);
      }
      set {
        _ExpReferenceIyy1ServerDataServerTime = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpReferenceIyy1ServerDataReferenceId
    /// </summary>
    private char _ExpReferenceIyy1ServerDataReferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public char ExpReferenceIyy1ServerDataReferenceId_AS {
      get {
        return(_ExpReferenceIyy1ServerDataReferenceId_AS);
      }
      set {
        _ExpReferenceIyy1ServerDataReferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpReferenceIyy1ServerDataReferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpReferenceIyy1ServerDataReferenceId;
    /// <summary>
    /// Attribute for: ExpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public string ExpReferenceIyy1ServerDataReferenceId {
      get {
        return(_ExpReferenceIyy1ServerDataReferenceId);
      }
      set {
        _ExpReferenceIyy1ServerDataReferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpReferenceIyy1ServerDataServerTimestamp
    /// </summary>
    private char _ExpReferenceIyy1ServerDataServerTimestamp_AS;
    /// <summary>
    /// Attribute missing flag for: ExpReferenceIyy1ServerDataServerTimestamp
    /// </summary>
    public char ExpReferenceIyy1ServerDataServerTimestamp_AS {
      get {
        return(_ExpReferenceIyy1ServerDataServerTimestamp_AS);
      }
      set {
        _ExpReferenceIyy1ServerDataServerTimestamp_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpReferenceIyy1ServerDataServerTimestamp
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpReferenceIyy1ServerDataServerTimestamp;
    /// <summary>
    /// Attribute for: ExpReferenceIyy1ServerDataServerTimestamp
    /// </summary>
    public string ExpReferenceIyy1ServerDataServerTimestamp {
      get {
        return(_ExpReferenceIyy1ServerDataServerTimestamp);
      }
      set {
        _ExpReferenceIyy1ServerDataServerTimestamp = value;
      }
    }
    // Entity View: EXP_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    private char _ExpErrorIyy1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public char ExpErrorIyy1ComponentSeverityCode_AS {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public string ExpErrorIyy1ComponentSeverityCode {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    private char _ExpErrorIyy1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public char ExpErrorIyy1ComponentRollbackIndicator_AS {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator_AS);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public string ExpErrorIyy1ComponentRollbackIndicator {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    private char _ExpErrorIyy1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public char ExpErrorIyy1ComponentOriginServid_AS {
      get {
        return(_ExpErrorIyy1ComponentOriginServid_AS);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ExpErrorIyy1ComponentOriginServid;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public double ExpErrorIyy1ComponentOriginServid {
      get {
        return(_ExpErrorIyy1ComponentOriginServid);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    private char _ExpErrorIyy1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public char ExpErrorIyy1ComponentContextString_AS {
      get {
        return(_ExpErrorIyy1ComponentContextString_AS);
      }
      set {
        _ExpErrorIyy1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _ExpErrorIyy1ComponentContextString;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public string ExpErrorIyy1ComponentContextString {
      get {
        return(_ExpErrorIyy1ComponentContextString);
      }
      set {
        _ExpErrorIyy1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public char ExpErrorIyy1ComponentReturnCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public int ExpErrorIyy1ComponentReturnCode {
      get {
        return(_ExpErrorIyy1ComponentReturnCode);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public char ExpErrorIyy1ComponentReasonCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public int ExpErrorIyy1ComponentReasonCode {
      get {
        return(_ExpErrorIyy1ComponentReasonCode);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    private char _ExpErrorIyy1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public char ExpErrorIyy1ComponentChecksum_AS {
      get {
        return(_ExpErrorIyy1ComponentChecksum_AS);
      }
      set {
        _ExpErrorIyy1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentChecksum;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public string ExpErrorIyy1ComponentChecksum {
      get {
        return(_ExpErrorIyy1ComponentChecksum);
      }
      set {
        _ExpErrorIyy1ComponentChecksum = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY97S1_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY97S1_OA( CYYY97S1_OA orig )
    {
      ExpReferenceIyy1ServerDataServerDate_AS = orig.ExpReferenceIyy1ServerDataServerDate_AS;
      ExpReferenceIyy1ServerDataServerDate = orig.ExpReferenceIyy1ServerDataServerDate;
      ExpReferenceIyy1ServerDataServerTime_AS = orig.ExpReferenceIyy1ServerDataServerTime_AS;
      ExpReferenceIyy1ServerDataServerTime = orig.ExpReferenceIyy1ServerDataServerTime;
      ExpReferenceIyy1ServerDataReferenceId_AS = orig.ExpReferenceIyy1ServerDataReferenceId_AS;
      ExpReferenceIyy1ServerDataReferenceId = orig.ExpReferenceIyy1ServerDataReferenceId;
      ExpReferenceIyy1ServerDataServerTimestamp_AS = orig.ExpReferenceIyy1ServerDataServerTimestamp_AS;
      ExpReferenceIyy1ServerDataServerTimestamp = orig.ExpReferenceIyy1ServerDataServerTimestamp;
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY97S1_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY97S1_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY97S1_OA());
          }
          else 
          {
            CYYY97S1_OA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new CYYY97S1_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpReferenceIyy1ServerDataServerDate_AS = ' ';
      ExpReferenceIyy1ServerDataServerDate = 00000000;
      ExpReferenceIyy1ServerDataServerTime_AS = ' ';
      ExpReferenceIyy1ServerDataServerTime = 00000000;
      ExpReferenceIyy1ServerDataReferenceId_AS = ' ';
      ExpReferenceIyy1ServerDataReferenceId = "00000000000000000000";
      ExpReferenceIyy1ServerDataServerTimestamp_AS = ' ';
      ExpReferenceIyy1ServerDataServerTimestamp = "00000000000000000000";
      ExpErrorIyy1ComponentSeverityCode_AS = ' ';
      ExpErrorIyy1ComponentSeverityCode = " ";
      ExpErrorIyy1ComponentRollbackIndicator_AS = ' ';
      ExpErrorIyy1ComponentRollbackIndicator = " ";
      ExpErrorIyy1ComponentOriginServid_AS = ' ';
      ExpErrorIyy1ComponentOriginServid = 0.0;
      ExpErrorIyy1ComponentContextString_AS = ' ';
      ExpErrorIyy1ComponentContextString = "";
      ExpErrorIyy1ComponentReturnCode_AS = ' ';
      ExpErrorIyy1ComponentReturnCode = 0;
      ExpErrorIyy1ComponentReasonCode_AS = ' ';
      ExpErrorIyy1ComponentReasonCode = 0;
      ExpErrorIyy1ComponentChecksum_AS = ' ';
      ExpErrorIyy1ComponentChecksum = "               ";
    }
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IExportView orig )
    {
      this.CopyFrom((CYYY97S1_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( CYYY97S1_OA orig )
    {
      ExpReferenceIyy1ServerDataServerDate_AS = orig.ExpReferenceIyy1ServerDataServerDate_AS;
      ExpReferenceIyy1ServerDataServerDate = orig.ExpReferenceIyy1ServerDataServerDate;
      ExpReferenceIyy1ServerDataServerTime_AS = orig.ExpReferenceIyy1ServerDataServerTime_AS;
      ExpReferenceIyy1ServerDataServerTime = orig.ExpReferenceIyy1ServerDataServerTime;
      ExpReferenceIyy1ServerDataReferenceId_AS = orig.ExpReferenceIyy1ServerDataReferenceId_AS;
      ExpReferenceIyy1ServerDataReferenceId = orig.ExpReferenceIyy1ServerDataReferenceId;
      ExpReferenceIyy1ServerDataServerTimestamp_AS = orig.ExpReferenceIyy1ServerDataServerTimestamp_AS;
      ExpReferenceIyy1ServerDataServerTimestamp = orig.ExpReferenceIyy1ServerDataServerTimestamp;
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
  }
}
