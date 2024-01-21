// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: DYYY0251_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:11
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
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: DYYY0251_IA
  /// </summary>
  [Serializable]
  public class DYYY0251_IA : ViewBase, IImportView
  {
    private static DYYY0251_IA[] freeArray = new DYYY0251_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentSeverityCode
    /// </summary>
    private char _ImpErrorIyy1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentSeverityCode
    /// </summary>
    public char ImpErrorIyy1ComponentSeverityCode_AS {
      get {
        return(_ImpErrorIyy1ComponentSeverityCode_AS);
      }
      set {
        _ImpErrorIyy1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIyy1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentSeverityCode
    /// </summary>
    public string ImpErrorIyy1ComponentSeverityCode {
      get {
        return(_ImpErrorIyy1ComponentSeverityCode);
      }
      set {
        _ImpErrorIyy1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    private char _ImpErrorIyy1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public char ImpErrorIyy1ComponentRollbackIndicator_AS {
      get {
        return(_ImpErrorIyy1ComponentRollbackIndicator_AS);
      }
      set {
        _ImpErrorIyy1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIyy1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public string ImpErrorIyy1ComponentRollbackIndicator {
      get {
        return(_ImpErrorIyy1ComponentRollbackIndicator);
      }
      set {
        _ImpErrorIyy1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentOriginServid
    /// </summary>
    private char _ImpErrorIyy1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentOriginServid
    /// </summary>
    public char ImpErrorIyy1ComponentOriginServid_AS {
      get {
        return(_ImpErrorIyy1ComponentOriginServid_AS);
      }
      set {
        _ImpErrorIyy1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ImpErrorIyy1ComponentOriginServid;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentOriginServid
    /// </summary>
    public double ImpErrorIyy1ComponentOriginServid {
      get {
        return(_ImpErrorIyy1ComponentOriginServid);
      }
      set {
        _ImpErrorIyy1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentContextString
    /// </summary>
    private char _ImpErrorIyy1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentContextString
    /// </summary>
    public char ImpErrorIyy1ComponentContextString_AS {
      get {
        return(_ImpErrorIyy1ComponentContextString_AS);
      }
      set {
        _ImpErrorIyy1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _ImpErrorIyy1ComponentContextString;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentContextString
    /// </summary>
    public string ImpErrorIyy1ComponentContextString {
      get {
        return(_ImpErrorIyy1ComponentContextString);
      }
      set {
        _ImpErrorIyy1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentReturnCode
    /// </summary>
    private char _ImpErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentReturnCode
    /// </summary>
    public char ImpErrorIyy1ComponentReturnCode_AS {
      get {
        return(_ImpErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _ImpErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentReturnCode
    /// </summary>
    public int ImpErrorIyy1ComponentReturnCode {
      get {
        return(_ImpErrorIyy1ComponentReturnCode);
      }
      set {
        _ImpErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentReasonCode
    /// </summary>
    private char _ImpErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentReasonCode
    /// </summary>
    public char ImpErrorIyy1ComponentReasonCode_AS {
      get {
        return(_ImpErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _ImpErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentReasonCode
    /// </summary>
    public int ImpErrorIyy1ComponentReasonCode {
      get {
        return(_ImpErrorIyy1ComponentReasonCode);
      }
      set {
        _ImpErrorIyy1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIyy1ComponentChecksum
    /// </summary>
    private char _ImpErrorIyy1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIyy1ComponentChecksum
    /// </summary>
    public char ImpErrorIyy1ComponentChecksum_AS {
      get {
        return(_ImpErrorIyy1ComponentChecksum_AS);
      }
      set {
        _ImpErrorIyy1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIyy1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIyy1ComponentChecksum;
    /// <summary>
    /// Attribute for: ImpErrorIyy1ComponentChecksum
    /// </summary>
    public string ImpErrorIyy1ComponentChecksum {
      get {
        return(_ImpErrorIyy1ComponentChecksum);
      }
      set {
        _ImpErrorIyy1ComponentChecksum = value;
      }
    }
    // Entity View: IMP_FILTER
    //        Type: IYY1_LIST
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    private char _ImpFilterIyy1ListSortOption_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    public char ImpFilterIyy1ListSortOption_AS {
      get {
        return(_ImpFilterIyy1ListSortOption_AS);
      }
      set {
        _ImpFilterIyy1ListSortOption_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListSortOption
    /// Domain: Text
    /// Length: 3
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListSortOption;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListSortOption
    /// </summary>
    public string ImpFilterIyy1ListSortOption {
      get {
        return(_ImpFilterIyy1ListSortOption);
      }
      set {
        _ImpFilterIyy1ListSortOption = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    private char _ImpFilterIyy1ListScrollType_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    public char ImpFilterIyy1ListScrollType_AS {
      get {
        return(_ImpFilterIyy1ListScrollType_AS);
      }
      set {
        _ImpFilterIyy1ListScrollType_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollType
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListScrollType;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollType
    /// </summary>
    public string ImpFilterIyy1ListScrollType {
      get {
        return(_ImpFilterIyy1ListScrollType);
      }
      set {
        _ImpFilterIyy1ListScrollType = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    private char _ImpFilterIyy1ListListDirection_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    public char ImpFilterIyy1ListListDirection_AS {
      get {
        return(_ImpFilterIyy1ListListDirection_AS);
      }
      set {
        _ImpFilterIyy1ListListDirection_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListListDirection
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListListDirection;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListListDirection
    /// </summary>
    public string ImpFilterIyy1ListListDirection {
      get {
        return(_ImpFilterIyy1ListListDirection);
      }
      set {
        _ImpFilterIyy1ListListDirection = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    private char _ImpFilterIyy1ListScrollAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public char ImpFilterIyy1ListScrollAmount_AS {
      get {
        return(_ImpFilterIyy1ListScrollAmount_AS);
      }
      set {
        _ImpFilterIyy1ListScrollAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollAmount
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterIyy1ListScrollAmount;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public int ImpFilterIyy1ListScrollAmount {
      get {
        return(_ImpFilterIyy1ListScrollAmount);
      }
      set {
        _ImpFilterIyy1ListScrollAmount = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    private char _ImpFilterIyy1ListOrderByFieldNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public char ImpFilterIyy1ListOrderByFieldNum_AS {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum_AS);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// Domain: Number
    /// Length: 1
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private short _ImpFilterIyy1ListOrderByFieldNum;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public short ImpFilterIyy1ListOrderByFieldNum {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum = value;
      }
    }
    // Entity View: IMP_FROM
    //        Type: CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromChildCinstanceId
    /// </summary>
    private char _ImpFromChildCinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromChildCinstanceId
    /// </summary>
    public char ImpFromChildCinstanceId_AS {
      get {
        return(_ImpFromChildCinstanceId_AS);
      }
      set {
        _ImpFromChildCinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromChildCinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpFromChildCinstanceId;
    /// <summary>
    /// Attribute for: ImpFromChildCinstanceId
    /// </summary>
    public string ImpFromChildCinstanceId {
      get {
        return(_ImpFromChildCinstanceId);
      }
      set {
        _ImpFromChildCinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFromChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromChildCparentPkeyAttrText
    /// </summary>
    public char ImpFromChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFromChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFromChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFromChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFromChildCparentPkeyAttrText
    /// </summary>
    public string ImpFromChildCparentPkeyAttrText {
      get {
        return(_ImpFromChildCparentPkeyAttrText);
      }
      set {
        _ImpFromChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromChildCkeyAttrNum
    /// </summary>
    private char _ImpFromChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromChildCkeyAttrNum
    /// </summary>
    public char ImpFromChildCkeyAttrNum_AS {
      get {
        return(_ImpFromChildCkeyAttrNum_AS);
      }
      set {
        _ImpFromChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFromChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFromChildCkeyAttrNum
    /// </summary>
    public int ImpFromChildCkeyAttrNum {
      get {
        return(_ImpFromChildCkeyAttrNum);
      }
      set {
        _ImpFromChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER_START
    //        Type: CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStartChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFilterStartChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStartChildCparentPkeyAttrText
    /// </summary>
    public char ImpFilterStartChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFilterStartChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFilterStartChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStartChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStartChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStartChildCparentPkeyAttrText
    /// </summary>
    public string ImpFilterStartChildCparentPkeyAttrText {
      get {
        return(_ImpFilterStartChildCparentPkeyAttrText);
      }
      set {
        _ImpFilterStartChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStartChildCkeyAttrNum
    /// </summary>
    private char _ImpFilterStartChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStartChildCkeyAttrNum
    /// </summary>
    public char ImpFilterStartChildCkeyAttrNum_AS {
      get {
        return(_ImpFilterStartChildCkeyAttrNum_AS);
      }
      set {
        _ImpFilterStartChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStartChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterStartChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFilterStartChildCkeyAttrNum
    /// </summary>
    public int ImpFilterStartChildCkeyAttrNum {
      get {
        return(_ImpFilterStartChildCkeyAttrNum);
      }
      set {
        _ImpFilterStartChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER_STOP
    //        Type: CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStopChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFilterStopChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStopChildCparentPkeyAttrText
    /// </summary>
    public char ImpFilterStopChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFilterStopChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFilterStopChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStopChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStopChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStopChildCparentPkeyAttrText
    /// </summary>
    public string ImpFilterStopChildCparentPkeyAttrText {
      get {
        return(_ImpFilterStopChildCparentPkeyAttrText);
      }
      set {
        _ImpFilterStopChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStopChildCkeyAttrNum
    /// </summary>
    private char _ImpFilterStopChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStopChildCkeyAttrNum
    /// </summary>
    public char ImpFilterStopChildCkeyAttrNum_AS {
      get {
        return(_ImpFilterStopChildCkeyAttrNum_AS);
      }
      set {
        _ImpFilterStopChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStopChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterStopChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFilterStopChildCkeyAttrNum
    /// </summary>
    public int ImpFilterStopChildCkeyAttrNum {
      get {
        return(_ImpFilterStopChildCkeyAttrNum);
      }
      set {
        _ImpFilterStopChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER
    //        Type: CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterChildCsearchAttrText
    /// </summary>
    private char _ImpFilterChildCsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterChildCsearchAttrText
    /// </summary>
    public char ImpFilterChildCsearchAttrText_AS {
      get {
        return(_ImpFilterChildCsearchAttrText_AS);
      }
      set {
        _ImpFilterChildCsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterChildCsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterChildCsearchAttrText;
    /// <summary>
    /// Attribute for: ImpFilterChildCsearchAttrText
    /// </summary>
    public string ImpFilterChildCsearchAttrText {
      get {
        return(_ImpFilterChildCsearchAttrText);
      }
      set {
        _ImpFilterChildCsearchAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public DYYY0251_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public DYYY0251_IA( DYYY0251_IA orig )
    {
      ImpErrorIyy1ComponentSeverityCode_AS = orig.ImpErrorIyy1ComponentSeverityCode_AS;
      ImpErrorIyy1ComponentSeverityCode = orig.ImpErrorIyy1ComponentSeverityCode;
      ImpErrorIyy1ComponentRollbackIndicator_AS = orig.ImpErrorIyy1ComponentRollbackIndicator_AS;
      ImpErrorIyy1ComponentRollbackIndicator = orig.ImpErrorIyy1ComponentRollbackIndicator;
      ImpErrorIyy1ComponentOriginServid_AS = orig.ImpErrorIyy1ComponentOriginServid_AS;
      ImpErrorIyy1ComponentOriginServid = orig.ImpErrorIyy1ComponentOriginServid;
      ImpErrorIyy1ComponentContextString_AS = orig.ImpErrorIyy1ComponentContextString_AS;
      ImpErrorIyy1ComponentContextString = orig.ImpErrorIyy1ComponentContextString;
      ImpErrorIyy1ComponentReturnCode_AS = orig.ImpErrorIyy1ComponentReturnCode_AS;
      ImpErrorIyy1ComponentReturnCode = orig.ImpErrorIyy1ComponentReturnCode;
      ImpErrorIyy1ComponentReasonCode_AS = orig.ImpErrorIyy1ComponentReasonCode_AS;
      ImpErrorIyy1ComponentReasonCode = orig.ImpErrorIyy1ComponentReasonCode;
      ImpErrorIyy1ComponentChecksum_AS = orig.ImpErrorIyy1ComponentChecksum_AS;
      ImpErrorIyy1ComponentChecksum = orig.ImpErrorIyy1ComponentChecksum;
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromChildCinstanceId_AS = orig.ImpFromChildCinstanceId_AS;
      ImpFromChildCinstanceId = orig.ImpFromChildCinstanceId;
      ImpFromChildCparentPkeyAttrText_AS = orig.ImpFromChildCparentPkeyAttrText_AS;
      ImpFromChildCparentPkeyAttrText = orig.ImpFromChildCparentPkeyAttrText;
      ImpFromChildCkeyAttrNum_AS = orig.ImpFromChildCkeyAttrNum_AS;
      ImpFromChildCkeyAttrNum = orig.ImpFromChildCkeyAttrNum;
      ImpFilterStartChildCparentPkeyAttrText_AS = orig.ImpFilterStartChildCparentPkeyAttrText_AS;
      ImpFilterStartChildCparentPkeyAttrText = orig.ImpFilterStartChildCparentPkeyAttrText;
      ImpFilterStartChildCkeyAttrNum_AS = orig.ImpFilterStartChildCkeyAttrNum_AS;
      ImpFilterStartChildCkeyAttrNum = orig.ImpFilterStartChildCkeyAttrNum;
      ImpFilterStopChildCparentPkeyAttrText_AS = orig.ImpFilterStopChildCparentPkeyAttrText_AS;
      ImpFilterStopChildCparentPkeyAttrText = orig.ImpFilterStopChildCparentPkeyAttrText;
      ImpFilterStopChildCkeyAttrNum_AS = orig.ImpFilterStopChildCkeyAttrNum_AS;
      ImpFilterStopChildCkeyAttrNum = orig.ImpFilterStopChildCkeyAttrNum;
      ImpFilterChildCsearchAttrText_AS = orig.ImpFilterChildCsearchAttrText_AS;
      ImpFilterChildCsearchAttrText = orig.ImpFilterChildCsearchAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static DYYY0251_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new DYYY0251_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new DYYY0251_IA());
          }
          else 
          {
            DYYY0251_IA result = freeArray[--countFree];
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
      return(new DYYY0251_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpErrorIyy1ComponentSeverityCode_AS = ' ';
      ImpErrorIyy1ComponentSeverityCode = " ";
      ImpErrorIyy1ComponentRollbackIndicator_AS = ' ';
      ImpErrorIyy1ComponentRollbackIndicator = " ";
      ImpErrorIyy1ComponentOriginServid_AS = ' ';
      ImpErrorIyy1ComponentOriginServid = 0.0;
      ImpErrorIyy1ComponentContextString_AS = ' ';
      ImpErrorIyy1ComponentContextString = "";
      ImpErrorIyy1ComponentReturnCode_AS = ' ';
      ImpErrorIyy1ComponentReturnCode = 0;
      ImpErrorIyy1ComponentReasonCode_AS = ' ';
      ImpErrorIyy1ComponentReasonCode = 0;
      ImpErrorIyy1ComponentChecksum_AS = ' ';
      ImpErrorIyy1ComponentChecksum = "               ";
      ImpFilterIyy1ListSortOption_AS = ' ';
      ImpFilterIyy1ListSortOption = "   ";
      ImpFilterIyy1ListScrollType_AS = ' ';
      ImpFilterIyy1ListScrollType = " ";
      ImpFilterIyy1ListListDirection_AS = ' ';
      ImpFilterIyy1ListListDirection = " ";
      ImpFilterIyy1ListScrollAmount_AS = ' ';
      ImpFilterIyy1ListScrollAmount = 0;
      ImpFilterIyy1ListOrderByFieldNum_AS = ' ';
      ImpFilterIyy1ListOrderByFieldNum = 0;
      ImpFromChildCinstanceId_AS = ' ';
      ImpFromChildCinstanceId = "00000000000000000000";
      ImpFromChildCparentPkeyAttrText_AS = ' ';
      ImpFromChildCparentPkeyAttrText = "     ";
      ImpFromChildCkeyAttrNum_AS = ' ';
      ImpFromChildCkeyAttrNum = 0;
      ImpFilterStartChildCparentPkeyAttrText_AS = ' ';
      ImpFilterStartChildCparentPkeyAttrText = "     ";
      ImpFilterStartChildCkeyAttrNum_AS = ' ';
      ImpFilterStartChildCkeyAttrNum = 0;
      ImpFilterStopChildCparentPkeyAttrText_AS = ' ';
      ImpFilterStopChildCparentPkeyAttrText = "     ";
      ImpFilterStopChildCkeyAttrNum_AS = ' ';
      ImpFilterStopChildCkeyAttrNum = 0;
      ImpFilterChildCsearchAttrText_AS = ' ';
      ImpFilterChildCsearchAttrText = "                         ";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((DYYY0251_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( DYYY0251_IA orig )
    {
      ImpErrorIyy1ComponentSeverityCode_AS = orig.ImpErrorIyy1ComponentSeverityCode_AS;
      ImpErrorIyy1ComponentSeverityCode = orig.ImpErrorIyy1ComponentSeverityCode;
      ImpErrorIyy1ComponentRollbackIndicator_AS = orig.ImpErrorIyy1ComponentRollbackIndicator_AS;
      ImpErrorIyy1ComponentRollbackIndicator = orig.ImpErrorIyy1ComponentRollbackIndicator;
      ImpErrorIyy1ComponentOriginServid_AS = orig.ImpErrorIyy1ComponentOriginServid_AS;
      ImpErrorIyy1ComponentOriginServid = orig.ImpErrorIyy1ComponentOriginServid;
      ImpErrorIyy1ComponentContextString_AS = orig.ImpErrorIyy1ComponentContextString_AS;
      ImpErrorIyy1ComponentContextString = orig.ImpErrorIyy1ComponentContextString;
      ImpErrorIyy1ComponentReturnCode_AS = orig.ImpErrorIyy1ComponentReturnCode_AS;
      ImpErrorIyy1ComponentReturnCode = orig.ImpErrorIyy1ComponentReturnCode;
      ImpErrorIyy1ComponentReasonCode_AS = orig.ImpErrorIyy1ComponentReasonCode_AS;
      ImpErrorIyy1ComponentReasonCode = orig.ImpErrorIyy1ComponentReasonCode;
      ImpErrorIyy1ComponentChecksum_AS = orig.ImpErrorIyy1ComponentChecksum_AS;
      ImpErrorIyy1ComponentChecksum = orig.ImpErrorIyy1ComponentChecksum;
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromChildCinstanceId_AS = orig.ImpFromChildCinstanceId_AS;
      ImpFromChildCinstanceId = orig.ImpFromChildCinstanceId;
      ImpFromChildCparentPkeyAttrText_AS = orig.ImpFromChildCparentPkeyAttrText_AS;
      ImpFromChildCparentPkeyAttrText = orig.ImpFromChildCparentPkeyAttrText;
      ImpFromChildCkeyAttrNum_AS = orig.ImpFromChildCkeyAttrNum_AS;
      ImpFromChildCkeyAttrNum = orig.ImpFromChildCkeyAttrNum;
      ImpFilterStartChildCparentPkeyAttrText_AS = orig.ImpFilterStartChildCparentPkeyAttrText_AS;
      ImpFilterStartChildCparentPkeyAttrText = orig.ImpFilterStartChildCparentPkeyAttrText;
      ImpFilterStartChildCkeyAttrNum_AS = orig.ImpFilterStartChildCkeyAttrNum_AS;
      ImpFilterStartChildCkeyAttrNum = orig.ImpFilterStartChildCkeyAttrNum;
      ImpFilterStopChildCparentPkeyAttrText_AS = orig.ImpFilterStopChildCparentPkeyAttrText_AS;
      ImpFilterStopChildCparentPkeyAttrText = orig.ImpFilterStopChildCparentPkeyAttrText;
      ImpFilterStopChildCkeyAttrNum_AS = orig.ImpFilterStopChildCkeyAttrNum_AS;
      ImpFilterStopChildCkeyAttrNum = orig.ImpFilterStopChildCkeyAttrNum;
      ImpFilterChildCsearchAttrText_AS = orig.ImpFilterChildCsearchAttrText_AS;
      ImpFilterChildCsearchAttrText = orig.ImpFilterChildCsearchAttrText;
    }
  }
}