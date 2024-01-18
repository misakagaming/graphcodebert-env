// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IYY10321_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:17
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
  /// Internal data view storage for: IYY10321_OA
  /// </summary>
  [Serializable]
  public class IYY10321_OA : ViewBase, IExportView
  {
    private static IYY10321_OA[] freeArray = new IYY10321_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTinstanceId
    /// </summary>
    private char _ExpIyy1TypeTinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTinstanceId
    /// </summary>
    public char ExpIyy1TypeTinstanceId_AS {
      get {
        return(_ExpIyy1TypeTinstanceId_AS);
      }
      set {
        _ExpIyy1TypeTinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpIyy1TypeTinstanceId;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTinstanceId
    /// </summary>
    public string ExpIyy1TypeTinstanceId {
      get {
        return(_ExpIyy1TypeTinstanceId);
      }
      set {
        _ExpIyy1TypeTinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTreferenceId
    /// </summary>
    private char _ExpIyy1TypeTreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTreferenceId
    /// </summary>
    public char ExpIyy1TypeTreferenceId_AS {
      get {
        return(_ExpIyy1TypeTreferenceId_AS);
      }
      set {
        _ExpIyy1TypeTreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpIyy1TypeTreferenceId;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTreferenceId
    /// </summary>
    public string ExpIyy1TypeTreferenceId {
      get {
        return(_ExpIyy1TypeTreferenceId);
      }
      set {
        _ExpIyy1TypeTreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTcreateUserId
    /// </summary>
    private char _ExpIyy1TypeTcreateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTcreateUserId
    /// </summary>
    public char ExpIyy1TypeTcreateUserId_AS {
      get {
        return(_ExpIyy1TypeTcreateUserId_AS);
      }
      set {
        _ExpIyy1TypeTcreateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTcreateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1TypeTcreateUserId;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTcreateUserId
    /// </summary>
    public string ExpIyy1TypeTcreateUserId {
      get {
        return(_ExpIyy1TypeTcreateUserId);
      }
      set {
        _ExpIyy1TypeTcreateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTupdateUserId
    /// </summary>
    private char _ExpIyy1TypeTupdateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTupdateUserId
    /// </summary>
    public char ExpIyy1TypeTupdateUserId_AS {
      get {
        return(_ExpIyy1TypeTupdateUserId_AS);
      }
      set {
        _ExpIyy1TypeTupdateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTupdateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1TypeTupdateUserId;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTupdateUserId
    /// </summary>
    public string ExpIyy1TypeTupdateUserId {
      get {
        return(_ExpIyy1TypeTupdateUserId);
      }
      set {
        _ExpIyy1TypeTupdateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTkeyAttrText
    /// </summary>
    private char _ExpIyy1TypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTkeyAttrText
    /// </summary>
    public char ExpIyy1TypeTkeyAttrText_AS {
      get {
        return(_ExpIyy1TypeTkeyAttrText_AS);
      }
      set {
        _ExpIyy1TypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1TypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTkeyAttrText
    /// </summary>
    public string ExpIyy1TypeTkeyAttrText {
      get {
        return(_ExpIyy1TypeTkeyAttrText);
      }
      set {
        _ExpIyy1TypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTsearchAttrText
    /// </summary>
    private char _ExpIyy1TypeTsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTsearchAttrText
    /// </summary>
    public char ExpIyy1TypeTsearchAttrText_AS {
      get {
        return(_ExpIyy1TypeTsearchAttrText_AS);
      }
      set {
        _ExpIyy1TypeTsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTsearchAttrText
    /// Domain: Text
    /// Length: 20
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1TypeTsearchAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTsearchAttrText
    /// </summary>
    public string ExpIyy1TypeTsearchAttrText {
      get {
        return(_ExpIyy1TypeTsearchAttrText);
      }
      set {
        _ExpIyy1TypeTsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTotherAttrText
    /// </summary>
    private char _ExpIyy1TypeTotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTotherAttrText
    /// </summary>
    public char ExpIyy1TypeTotherAttrText_AS {
      get {
        return(_ExpIyy1TypeTotherAttrText_AS);
      }
      set {
        _ExpIyy1TypeTotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTotherAttrText
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1TypeTotherAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTotherAttrText
    /// </summary>
    public string ExpIyy1TypeTotherAttrText {
      get {
        return(_ExpIyy1TypeTotherAttrText);
      }
      set {
        _ExpIyy1TypeTotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTotherAttrDate
    /// </summary>
    private char _ExpIyy1TypeTotherAttrDate_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTotherAttrDate
    /// </summary>
    public char ExpIyy1TypeTotherAttrDate_AS {
      get {
        return(_ExpIyy1TypeTotherAttrDate_AS);
      }
      set {
        _ExpIyy1TypeTotherAttrDate_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTotherAttrDate
    /// Domain: Date
    /// Length: 8
    /// </summary>
    private int _ExpIyy1TypeTotherAttrDate;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTotherAttrDate
    /// </summary>
    public int ExpIyy1TypeTotherAttrDate {
      get {
        return(_ExpIyy1TypeTotherAttrDate);
      }
      set {
        _ExpIyy1TypeTotherAttrDate = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTotherAttrTime
    /// </summary>
    private char _ExpIyy1TypeTotherAttrTime_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTotherAttrTime
    /// </summary>
    public char ExpIyy1TypeTotherAttrTime_AS {
      get {
        return(_ExpIyy1TypeTotherAttrTime_AS);
      }
      set {
        _ExpIyy1TypeTotherAttrTime_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTotherAttrTime
    /// Domain: Time
    /// Length: 6
    /// </summary>
    private int _ExpIyy1TypeTotherAttrTime;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTotherAttrTime
    /// </summary>
    public int ExpIyy1TypeTotherAttrTime {
      get {
        return(_ExpIyy1TypeTotherAttrTime);
      }
      set {
        _ExpIyy1TypeTotherAttrTime = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1TypeTotherAttrAmount
    /// </summary>
    private char _ExpIyy1TypeTotherAttrAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1TypeTotherAttrAmount
    /// </summary>
    public char ExpIyy1TypeTotherAttrAmount_AS {
      get {
        return(_ExpIyy1TypeTotherAttrAmount_AS);
      }
      set {
        _ExpIyy1TypeTotherAttrAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1TypeTotherAttrAmount
    /// Domain: Number
    /// Length: 17
    /// Decimal Places: 2
    /// Decimal Precision: Y
    /// </summary>
    private decimal _ExpIyy1TypeTotherAttrAmount;
    /// <summary>
    /// Attribute for: ExpIyy1TypeTotherAttrAmount
    /// </summary>
    public decimal ExpIyy1TypeTotherAttrAmount {
      get {
        return(_ExpIyy1TypeTotherAttrAmount);
      }
      set {
        _ExpIyy1TypeTotherAttrAmount = value;
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
    
    public IYY10321_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IYY10321_OA( IYY10321_OA orig )
    {
      ExpIyy1TypeTinstanceId_AS = orig.ExpIyy1TypeTinstanceId_AS;
      ExpIyy1TypeTinstanceId = orig.ExpIyy1TypeTinstanceId;
      ExpIyy1TypeTreferenceId_AS = orig.ExpIyy1TypeTreferenceId_AS;
      ExpIyy1TypeTreferenceId = orig.ExpIyy1TypeTreferenceId;
      ExpIyy1TypeTcreateUserId_AS = orig.ExpIyy1TypeTcreateUserId_AS;
      ExpIyy1TypeTcreateUserId = orig.ExpIyy1TypeTcreateUserId;
      ExpIyy1TypeTupdateUserId_AS = orig.ExpIyy1TypeTupdateUserId_AS;
      ExpIyy1TypeTupdateUserId = orig.ExpIyy1TypeTupdateUserId;
      ExpIyy1TypeTkeyAttrText_AS = orig.ExpIyy1TypeTkeyAttrText_AS;
      ExpIyy1TypeTkeyAttrText = orig.ExpIyy1TypeTkeyAttrText;
      ExpIyy1TypeTsearchAttrText_AS = orig.ExpIyy1TypeTsearchAttrText_AS;
      ExpIyy1TypeTsearchAttrText = orig.ExpIyy1TypeTsearchAttrText;
      ExpIyy1TypeTotherAttrText_AS = orig.ExpIyy1TypeTotherAttrText_AS;
      ExpIyy1TypeTotherAttrText = orig.ExpIyy1TypeTotherAttrText;
      ExpIyy1TypeTotherAttrDate_AS = orig.ExpIyy1TypeTotherAttrDate_AS;
      ExpIyy1TypeTotherAttrDate = orig.ExpIyy1TypeTotherAttrDate;
      ExpIyy1TypeTotherAttrTime_AS = orig.ExpIyy1TypeTotherAttrTime_AS;
      ExpIyy1TypeTotherAttrTime = orig.ExpIyy1TypeTotherAttrTime;
      ExpIyy1TypeTotherAttrAmount_AS = orig.ExpIyy1TypeTotherAttrAmount_AS;
      ExpIyy1TypeTotherAttrAmount = orig.ExpIyy1TypeTotherAttrAmount;
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
    
    public static IYY10321_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IYY10321_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IYY10321_OA());
          }
          else 
          {
            IYY10321_OA result = freeArray[--countFree];
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
      return(new IYY10321_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpIyy1TypeTinstanceId_AS = ' ';
      ExpIyy1TypeTinstanceId = "00000000000000000000";
      ExpIyy1TypeTreferenceId_AS = ' ';
      ExpIyy1TypeTreferenceId = "00000000000000000000";
      ExpIyy1TypeTcreateUserId_AS = ' ';
      ExpIyy1TypeTcreateUserId = "        ";
      ExpIyy1TypeTupdateUserId_AS = ' ';
      ExpIyy1TypeTupdateUserId = "        ";
      ExpIyy1TypeTkeyAttrText_AS = ' ';
      ExpIyy1TypeTkeyAttrText = "    ";
      ExpIyy1TypeTsearchAttrText_AS = ' ';
      ExpIyy1TypeTsearchAttrText = "                    ";
      ExpIyy1TypeTotherAttrText_AS = ' ';
      ExpIyy1TypeTotherAttrText = "  ";
      ExpIyy1TypeTotherAttrDate_AS = ' ';
      ExpIyy1TypeTotherAttrDate = 00000000;
      ExpIyy1TypeTotherAttrTime_AS = ' ';
      ExpIyy1TypeTotherAttrTime = 00000000;
      ExpIyy1TypeTotherAttrAmount_AS = ' ';
      ExpIyy1TypeTotherAttrAmount = DecimalAttr.GetDefaultValue();
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
      this.CopyFrom((IYY10321_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IYY10321_OA orig )
    {
      ExpIyy1TypeTinstanceId_AS = orig.ExpIyy1TypeTinstanceId_AS;
      ExpIyy1TypeTinstanceId = orig.ExpIyy1TypeTinstanceId;
      ExpIyy1TypeTreferenceId_AS = orig.ExpIyy1TypeTreferenceId_AS;
      ExpIyy1TypeTreferenceId = orig.ExpIyy1TypeTreferenceId;
      ExpIyy1TypeTcreateUserId_AS = orig.ExpIyy1TypeTcreateUserId_AS;
      ExpIyy1TypeTcreateUserId = orig.ExpIyy1TypeTcreateUserId;
      ExpIyy1TypeTupdateUserId_AS = orig.ExpIyy1TypeTupdateUserId_AS;
      ExpIyy1TypeTupdateUserId = orig.ExpIyy1TypeTupdateUserId;
      ExpIyy1TypeTkeyAttrText_AS = orig.ExpIyy1TypeTkeyAttrText_AS;
      ExpIyy1TypeTkeyAttrText = orig.ExpIyy1TypeTkeyAttrText;
      ExpIyy1TypeTsearchAttrText_AS = orig.ExpIyy1TypeTsearchAttrText_AS;
      ExpIyy1TypeTsearchAttrText = orig.ExpIyy1TypeTsearchAttrText;
      ExpIyy1TypeTotherAttrText_AS = orig.ExpIyy1TypeTotherAttrText_AS;
      ExpIyy1TypeTotherAttrText = orig.ExpIyy1TypeTotherAttrText;
      ExpIyy1TypeTotherAttrDate_AS = orig.ExpIyy1TypeTotherAttrDate_AS;
      ExpIyy1TypeTotherAttrDate = orig.ExpIyy1TypeTotherAttrDate;
      ExpIyy1TypeTotherAttrTime_AS = orig.ExpIyy1TypeTotherAttrTime_AS;
      ExpIyy1TypeTotherAttrTime = orig.ExpIyy1TypeTotherAttrTime;
      ExpIyy1TypeTotherAttrAmount_AS = orig.ExpIyy1TypeTotherAttrAmount_AS;
      ExpIyy1TypeTotherAttrAmount = orig.ExpIyy1TypeTotherAttrAmount;
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
