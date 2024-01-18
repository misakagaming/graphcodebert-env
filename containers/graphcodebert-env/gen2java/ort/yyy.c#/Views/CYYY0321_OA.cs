// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY0321_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:13
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
  /// Internal data view storage for: CYYY0321_OA
  /// </summary>
  [Serializable]
  public class CYYY0321_OA : ViewBase, IExportView
  {
    private static CYYY0321_OA[] freeArray = new CYYY0321_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP
    //        Type: TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTinstanceId
    /// </summary>
    private char _ExpTypeTinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTinstanceId
    /// </summary>
    public char ExpTypeTinstanceId_AS {
      get {
        return(_ExpTypeTinstanceId_AS);
      }
      set {
        _ExpTypeTinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpTypeTinstanceId;
    /// <summary>
    /// Attribute for: ExpTypeTinstanceId
    /// </summary>
    public string ExpTypeTinstanceId {
      get {
        return(_ExpTypeTinstanceId);
      }
      set {
        _ExpTypeTinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTreferenceId
    /// </summary>
    private char _ExpTypeTreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTreferenceId
    /// </summary>
    public char ExpTypeTreferenceId_AS {
      get {
        return(_ExpTypeTreferenceId_AS);
      }
      set {
        _ExpTypeTreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpTypeTreferenceId;
    /// <summary>
    /// Attribute for: ExpTypeTreferenceId
    /// </summary>
    public string ExpTypeTreferenceId {
      get {
        return(_ExpTypeTreferenceId);
      }
      set {
        _ExpTypeTreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTcreateUserId
    /// </summary>
    private char _ExpTypeTcreateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTcreateUserId
    /// </summary>
    public char ExpTypeTcreateUserId_AS {
      get {
        return(_ExpTypeTcreateUserId_AS);
      }
      set {
        _ExpTypeTcreateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTcreateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpTypeTcreateUserId;
    /// <summary>
    /// Attribute for: ExpTypeTcreateUserId
    /// </summary>
    public string ExpTypeTcreateUserId {
      get {
        return(_ExpTypeTcreateUserId);
      }
      set {
        _ExpTypeTcreateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTupdateUserId
    /// </summary>
    private char _ExpTypeTupdateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTupdateUserId
    /// </summary>
    public char ExpTypeTupdateUserId_AS {
      get {
        return(_ExpTypeTupdateUserId_AS);
      }
      set {
        _ExpTypeTupdateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTupdateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpTypeTupdateUserId;
    /// <summary>
    /// Attribute for: ExpTypeTupdateUserId
    /// </summary>
    public string ExpTypeTupdateUserId {
      get {
        return(_ExpTypeTupdateUserId);
      }
      set {
        _ExpTypeTupdateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTkeyAttrText
    /// </summary>
    private char _ExpTypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTkeyAttrText
    /// </summary>
    public char ExpTypeTkeyAttrText_AS {
      get {
        return(_ExpTypeTkeyAttrText_AS);
      }
      set {
        _ExpTypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ExpTypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ExpTypeTkeyAttrText
    /// </summary>
    public string ExpTypeTkeyAttrText {
      get {
        return(_ExpTypeTkeyAttrText);
      }
      set {
        _ExpTypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTsearchAttrText
    /// </summary>
    private char _ExpTypeTsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTsearchAttrText
    /// </summary>
    public char ExpTypeTsearchAttrText_AS {
      get {
        return(_ExpTypeTsearchAttrText_AS);
      }
      set {
        _ExpTypeTsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTsearchAttrText
    /// Domain: Text
    /// Length: 20
    /// Varying Length: N
    /// </summary>
    private string _ExpTypeTsearchAttrText;
    /// <summary>
    /// Attribute for: ExpTypeTsearchAttrText
    /// </summary>
    public string ExpTypeTsearchAttrText {
      get {
        return(_ExpTypeTsearchAttrText);
      }
      set {
        _ExpTypeTsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTotherAttrText
    /// </summary>
    private char _ExpTypeTotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTotherAttrText
    /// </summary>
    public char ExpTypeTotherAttrText_AS {
      get {
        return(_ExpTypeTotherAttrText_AS);
      }
      set {
        _ExpTypeTotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTotherAttrText
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ExpTypeTotherAttrText;
    /// <summary>
    /// Attribute for: ExpTypeTotherAttrText
    /// </summary>
    public string ExpTypeTotherAttrText {
      get {
        return(_ExpTypeTotherAttrText);
      }
      set {
        _ExpTypeTotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTotherAttrDate
    /// </summary>
    private char _ExpTypeTotherAttrDate_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTotherAttrDate
    /// </summary>
    public char ExpTypeTotherAttrDate_AS {
      get {
        return(_ExpTypeTotherAttrDate_AS);
      }
      set {
        _ExpTypeTotherAttrDate_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTotherAttrDate
    /// Domain: Date
    /// Length: 8
    /// </summary>
    private int _ExpTypeTotherAttrDate;
    /// <summary>
    /// Attribute for: ExpTypeTotherAttrDate
    /// </summary>
    public int ExpTypeTotherAttrDate {
      get {
        return(_ExpTypeTotherAttrDate);
      }
      set {
        _ExpTypeTotherAttrDate = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTotherAttrTime
    /// </summary>
    private char _ExpTypeTotherAttrTime_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTotherAttrTime
    /// </summary>
    public char ExpTypeTotherAttrTime_AS {
      get {
        return(_ExpTypeTotherAttrTime_AS);
      }
      set {
        _ExpTypeTotherAttrTime_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTotherAttrTime
    /// Domain: Time
    /// Length: 6
    /// </summary>
    private int _ExpTypeTotherAttrTime;
    /// <summary>
    /// Attribute for: ExpTypeTotherAttrTime
    /// </summary>
    public int ExpTypeTotherAttrTime {
      get {
        return(_ExpTypeTotherAttrTime);
      }
      set {
        _ExpTypeTotherAttrTime = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTotherAttrAmount
    /// </summary>
    private char _ExpTypeTotherAttrAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTotherAttrAmount
    /// </summary>
    public char ExpTypeTotherAttrAmount_AS {
      get {
        return(_ExpTypeTotherAttrAmount_AS);
      }
      set {
        _ExpTypeTotherAttrAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTotherAttrAmount
    /// Domain: Number
    /// Length: 17
    /// Decimal Places: 2
    /// Decimal Precision: Y
    /// </summary>
    private decimal _ExpTypeTotherAttrAmount;
    /// <summary>
    /// Attribute for: ExpTypeTotherAttrAmount
    /// </summary>
    public decimal ExpTypeTotherAttrAmount {
      get {
        return(_ExpTypeTotherAttrAmount);
      }
      set {
        _ExpTypeTotherAttrAmount = value;
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
    
    public CYYY0321_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY0321_OA( CYYY0321_OA orig )
    {
      ExpTypeTinstanceId_AS = orig.ExpTypeTinstanceId_AS;
      ExpTypeTinstanceId = orig.ExpTypeTinstanceId;
      ExpTypeTreferenceId_AS = orig.ExpTypeTreferenceId_AS;
      ExpTypeTreferenceId = orig.ExpTypeTreferenceId;
      ExpTypeTcreateUserId_AS = orig.ExpTypeTcreateUserId_AS;
      ExpTypeTcreateUserId = orig.ExpTypeTcreateUserId;
      ExpTypeTupdateUserId_AS = orig.ExpTypeTupdateUserId_AS;
      ExpTypeTupdateUserId = orig.ExpTypeTupdateUserId;
      ExpTypeTkeyAttrText_AS = orig.ExpTypeTkeyAttrText_AS;
      ExpTypeTkeyAttrText = orig.ExpTypeTkeyAttrText;
      ExpTypeTsearchAttrText_AS = orig.ExpTypeTsearchAttrText_AS;
      ExpTypeTsearchAttrText = orig.ExpTypeTsearchAttrText;
      ExpTypeTotherAttrText_AS = orig.ExpTypeTotherAttrText_AS;
      ExpTypeTotherAttrText = orig.ExpTypeTotherAttrText;
      ExpTypeTotherAttrDate_AS = orig.ExpTypeTotherAttrDate_AS;
      ExpTypeTotherAttrDate = orig.ExpTypeTotherAttrDate;
      ExpTypeTotherAttrTime_AS = orig.ExpTypeTotherAttrTime_AS;
      ExpTypeTotherAttrTime = orig.ExpTypeTotherAttrTime;
      ExpTypeTotherAttrAmount_AS = orig.ExpTypeTotherAttrAmount_AS;
      ExpTypeTotherAttrAmount = orig.ExpTypeTotherAttrAmount;
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
    
    public static CYYY0321_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY0321_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY0321_OA());
          }
          else 
          {
            CYYY0321_OA result = freeArray[--countFree];
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
      return(new CYYY0321_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpTypeTinstanceId_AS = ' ';
      ExpTypeTinstanceId = "00000000000000000000";
      ExpTypeTreferenceId_AS = ' ';
      ExpTypeTreferenceId = "00000000000000000000";
      ExpTypeTcreateUserId_AS = ' ';
      ExpTypeTcreateUserId = "        ";
      ExpTypeTupdateUserId_AS = ' ';
      ExpTypeTupdateUserId = "        ";
      ExpTypeTkeyAttrText_AS = ' ';
      ExpTypeTkeyAttrText = "    ";
      ExpTypeTsearchAttrText_AS = ' ';
      ExpTypeTsearchAttrText = "                    ";
      ExpTypeTotherAttrText_AS = ' ';
      ExpTypeTotherAttrText = "  ";
      ExpTypeTotherAttrDate_AS = ' ';
      ExpTypeTotherAttrDate = 00000000;
      ExpTypeTotherAttrTime_AS = ' ';
      ExpTypeTotherAttrTime = 00000000;
      ExpTypeTotherAttrAmount_AS = ' ';
      ExpTypeTotherAttrAmount = DecimalAttr.GetDefaultValue();
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
      this.CopyFrom((CYYY0321_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( CYYY0321_OA orig )
    {
      ExpTypeTinstanceId_AS = orig.ExpTypeTinstanceId_AS;
      ExpTypeTinstanceId = orig.ExpTypeTinstanceId;
      ExpTypeTreferenceId_AS = orig.ExpTypeTreferenceId_AS;
      ExpTypeTreferenceId = orig.ExpTypeTreferenceId;
      ExpTypeTcreateUserId_AS = orig.ExpTypeTcreateUserId_AS;
      ExpTypeTcreateUserId = orig.ExpTypeTcreateUserId;
      ExpTypeTupdateUserId_AS = orig.ExpTypeTupdateUserId_AS;
      ExpTypeTupdateUserId = orig.ExpTypeTupdateUserId;
      ExpTypeTkeyAttrText_AS = orig.ExpTypeTkeyAttrText_AS;
      ExpTypeTkeyAttrText = orig.ExpTypeTkeyAttrText;
      ExpTypeTsearchAttrText_AS = orig.ExpTypeTsearchAttrText_AS;
      ExpTypeTsearchAttrText = orig.ExpTypeTsearchAttrText;
      ExpTypeTotherAttrText_AS = orig.ExpTypeTotherAttrText_AS;
      ExpTypeTotherAttrText = orig.ExpTypeTotherAttrText;
      ExpTypeTotherAttrDate_AS = orig.ExpTypeTotherAttrDate_AS;
      ExpTypeTotherAttrDate = orig.ExpTypeTotherAttrDate;
      ExpTypeTotherAttrTime_AS = orig.ExpTypeTotherAttrTime_AS;
      ExpTypeTotherAttrTime = orig.ExpTypeTotherAttrTime;
      ExpTypeTotherAttrAmount_AS = orig.ExpTypeTotherAttrAmount_AS;
      ExpTypeTotherAttrAmount = orig.ExpTypeTotherAttrAmount;
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
