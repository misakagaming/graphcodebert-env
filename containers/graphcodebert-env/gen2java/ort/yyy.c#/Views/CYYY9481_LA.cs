// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY9481_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:47
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
// START OF LOCAL VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: CYYY9481_LA
  /// </summary>
  [Serializable]
  public class CYYY9481_LA : ViewBase, ILocalView
  {
    private static CYYY9481_LA[] freeArray = new CYYY9481_LA[30];
    private static int countFree = 0;
    
    // Entity View: LOC_IMP_ERROR
    //        Type: IRO1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentSpecificationId
    /// </summary>
    private char _LocImpErrorIro1ComponentSpecificationId_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentSpecificationId
    /// </summary>
    public char LocImpErrorIro1ComponentSpecificationId_AS {
      get {
        return(_LocImpErrorIro1ComponentSpecificationId_AS);
      }
      set {
        _LocImpErrorIro1ComponentSpecificationId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentSpecificationId
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _LocImpErrorIro1ComponentSpecificationId;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentSpecificationId
    /// </summary>
    public double LocImpErrorIro1ComponentSpecificationId {
      get {
        return(_LocImpErrorIro1ComponentSpecificationId);
      }
      set {
        _LocImpErrorIro1ComponentSpecificationId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentImplementationId
    /// </summary>
    private char _LocImpErrorIro1ComponentImplementationId_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentImplementationId
    /// </summary>
    public char LocImpErrorIro1ComponentImplementationId_AS {
      get {
        return(_LocImpErrorIro1ComponentImplementationId_AS);
      }
      set {
        _LocImpErrorIro1ComponentImplementationId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentImplementationId
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _LocImpErrorIro1ComponentImplementationId;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentImplementationId
    /// </summary>
    public double LocImpErrorIro1ComponentImplementationId {
      get {
        return(_LocImpErrorIro1ComponentImplementationId);
      }
      set {
        _LocImpErrorIro1ComponentImplementationId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentOriginServid
    /// </summary>
    private char _LocImpErrorIro1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentOriginServid
    /// </summary>
    public char LocImpErrorIro1ComponentOriginServid_AS {
      get {
        return(_LocImpErrorIro1ComponentOriginServid_AS);
      }
      set {
        _LocImpErrorIro1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _LocImpErrorIro1ComponentOriginServid;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentOriginServid
    /// </summary>
    public double LocImpErrorIro1ComponentOriginServid {
      get {
        return(_LocImpErrorIro1ComponentOriginServid);
      }
      set {
        _LocImpErrorIro1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentReturnCode
    /// </summary>
    private char _LocImpErrorIro1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentReturnCode
    /// </summary>
    public char LocImpErrorIro1ComponentReturnCode_AS {
      get {
        return(_LocImpErrorIro1ComponentReturnCode_AS);
      }
      set {
        _LocImpErrorIro1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocImpErrorIro1ComponentReturnCode;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentReturnCode
    /// </summary>
    public int LocImpErrorIro1ComponentReturnCode {
      get {
        return(_LocImpErrorIro1ComponentReturnCode);
      }
      set {
        _LocImpErrorIro1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentReasonCode
    /// </summary>
    private char _LocImpErrorIro1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentReasonCode
    /// </summary>
    public char LocImpErrorIro1ComponentReasonCode_AS {
      get {
        return(_LocImpErrorIro1ComponentReasonCode_AS);
      }
      set {
        _LocImpErrorIro1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocImpErrorIro1ComponentReasonCode;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentReasonCode
    /// </summary>
    public int LocImpErrorIro1ComponentReasonCode {
      get {
        return(_LocImpErrorIro1ComponentReasonCode);
      }
      set {
        _LocImpErrorIro1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentContextStringTx
    /// </summary>
    private char _LocImpErrorIro1ComponentContextStringTx_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentContextStringTx
    /// </summary>
    public char LocImpErrorIro1ComponentContextStringTx_AS {
      get {
        return(_LocImpErrorIro1ComponentContextStringTx_AS);
      }
      set {
        _LocImpErrorIro1ComponentContextStringTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentContextStringTx
    /// Domain: Text
    /// Length: 255
    /// Varying Length: Y
    /// </summary>
    private string _LocImpErrorIro1ComponentContextStringTx;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentContextStringTx
    /// </summary>
    public string LocImpErrorIro1ComponentContextStringTx {
      get {
        return(_LocImpErrorIro1ComponentContextStringTx);
      }
      set {
        _LocImpErrorIro1ComponentContextStringTx = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentActivityCd
    /// </summary>
    private char _LocImpErrorIro1ComponentActivityCd_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentActivityCd
    /// </summary>
    public char LocImpErrorIro1ComponentActivityCd_AS {
      get {
        return(_LocImpErrorIro1ComponentActivityCd_AS);
      }
      set {
        _LocImpErrorIro1ComponentActivityCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentActivityCd
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _LocImpErrorIro1ComponentActivityCd;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentActivityCd
    /// </summary>
    public string LocImpErrorIro1ComponentActivityCd {
      get {
        return(_LocImpErrorIro1ComponentActivityCd);
      }
      set {
        _LocImpErrorIro1ComponentActivityCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpErrorIro1ComponentDialectCd
    /// </summary>
    private char _LocImpErrorIro1ComponentDialectCd_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpErrorIro1ComponentDialectCd
    /// </summary>
    public char LocImpErrorIro1ComponentDialectCd_AS {
      get {
        return(_LocImpErrorIro1ComponentDialectCd_AS);
      }
      set {
        _LocImpErrorIro1ComponentDialectCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpErrorIro1ComponentDialectCd
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _LocImpErrorIro1ComponentDialectCd;
    /// <summary>
    /// Attribute for: LocImpErrorIro1ComponentDialectCd
    /// </summary>
    public string LocImpErrorIro1ComponentDialectCd {
      get {
        return(_LocImpErrorIro1ComponentDialectCd);
      }
      set {
        _LocImpErrorIro1ComponentDialectCd = value;
      }
    }
    // Repeating GV:  LOC_EXP_HILITE_GROUP
    //     Repeats: 10 times
    /// <summary>
    /// Internal storage, repeating group view count
    /// </summary>
    private int _LocExpHiliteGroup_MA;
    /// <summary>
    /// Repeating group view count
    /// </summary>
    public int LocExpHiliteGroup_MA {
      get {
        return(_LocExpHiliteGroup_MA);
      }
      set {
        _LocExpHiliteGroup_MA = value;
      }
    }
    /// <summary>
    /// Internal storage, repeating group view occurrance array
    /// </summary>
    private char[] _LocExpHiliteGroup_AC = new char[10];
    /// <summary>
    /// Repeating group view occurrance array
    /// </summary>
    public char[] LocExpHiliteGroup_AC {
      get {
        return(_LocExpHiliteGroup_AC);
      }
      set {
        _LocExpHiliteGroup_AC = value;
      }
    }
    // Entity View: LOC_EXP_HILITE_G
    //        Type: IRO1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpHiliteGIro1ComponentAttributeNameTx
    /// </summary>
    private char[] _LocExpHiliteGIro1ComponentAttributeNameTx_AS = new char[10];
    /// <summary>
    /// Attribute missing flag for: LocExpHiliteGIro1ComponentAttributeNameTx
    /// </summary>
    public char[] LocExpHiliteGIro1ComponentAttributeNameTx_AS {
      get {
        return(_LocExpHiliteGIro1ComponentAttributeNameTx_AS);
      }
      set {
        _LocExpHiliteGIro1ComponentAttributeNameTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpHiliteGIro1ComponentAttributeNameTx
    /// Domain: Text
    /// Length: 50
    /// Varying Length: N
    /// </summary>
    private string[] _LocExpHiliteGIro1ComponentAttributeNameTx = new string[10];
    /// <summary>
    /// Attribute for: LocExpHiliteGIro1ComponentAttributeNameTx
    /// </summary>
    public string[] LocExpHiliteGIro1ComponentAttributeNameTx {
      get {
        return(_LocExpHiliteGIro1ComponentAttributeNameTx);
      }
      set {
        _LocExpHiliteGIro1ComponentAttributeNameTx = value;
      }
    }
    // Entity View: LOC_OTHER_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocOtherErrorIyy1ComponentReturnCode
    /// </summary>
    private char _LocOtherErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocOtherErrorIyy1ComponentReturnCode
    /// </summary>
    public char LocOtherErrorIyy1ComponentReturnCode_AS {
      get {
        return(_LocOtherErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _LocOtherErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocOtherErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocOtherErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: LocOtherErrorIyy1ComponentReturnCode
    /// </summary>
    public int LocOtherErrorIyy1ComponentReturnCode {
      get {
        return(_LocOtherErrorIyy1ComponentReturnCode);
      }
      set {
        _LocOtherErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocOtherErrorIyy1ComponentReasonCode
    /// </summary>
    private char _LocOtherErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocOtherErrorIyy1ComponentReasonCode
    /// </summary>
    public char LocOtherErrorIyy1ComponentReasonCode_AS {
      get {
        return(_LocOtherErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _LocOtherErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocOtherErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocOtherErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: LocOtherErrorIyy1ComponentReasonCode
    /// </summary>
    public int LocOtherErrorIyy1ComponentReasonCode {
      get {
        return(_LocOtherErrorIyy1ComponentReasonCode);
      }
      set {
        _LocOtherErrorIyy1ComponentReasonCode = value;
      }
    }
    // Entity View: LOC_ERROR_MSG
    //        Type: IRO1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorMsgIro1ComponentSeverityCd
    /// </summary>
    private char _LocErrorMsgIro1ComponentSeverityCd_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorMsgIro1ComponentSeverityCd
    /// </summary>
    public char LocErrorMsgIro1ComponentSeverityCd_AS {
      get {
        return(_LocErrorMsgIro1ComponentSeverityCd_AS);
      }
      set {
        _LocErrorMsgIro1ComponentSeverityCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorMsgIro1ComponentSeverityCd
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _LocErrorMsgIro1ComponentSeverityCd;
    /// <summary>
    /// Attribute for: LocErrorMsgIro1ComponentSeverityCd
    /// </summary>
    public string LocErrorMsgIro1ComponentSeverityCd {
      get {
        return(_LocErrorMsgIro1ComponentSeverityCd);
      }
      set {
        _LocErrorMsgIro1ComponentSeverityCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorMsgIro1ComponentMessageTx
    /// </summary>
    private char _LocErrorMsgIro1ComponentMessageTx_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorMsgIro1ComponentMessageTx
    /// </summary>
    public char LocErrorMsgIro1ComponentMessageTx_AS {
      get {
        return(_LocErrorMsgIro1ComponentMessageTx_AS);
      }
      set {
        _LocErrorMsgIro1ComponentMessageTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorMsgIro1ComponentMessageTx
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _LocErrorMsgIro1ComponentMessageTx;
    /// <summary>
    /// Attribute for: LocErrorMsgIro1ComponentMessageTx
    /// </summary>
    public string LocErrorMsgIro1ComponentMessageTx {
      get {
        return(_LocErrorMsgIro1ComponentMessageTx);
      }
      set {
        _LocErrorMsgIro1ComponentMessageTx = value;
      }
    }
    // Entity View: LOC_ERROR
    //        Type: IRO1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentOriginServid
    /// </summary>
    private char _LocErrorIro1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentOriginServid
    /// </summary>
    public char LocErrorIro1ComponentOriginServid_AS {
      get {
        return(_LocErrorIro1ComponentOriginServid_AS);
      }
      set {
        _LocErrorIro1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _LocErrorIro1ComponentOriginServid;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentOriginServid
    /// </summary>
    public double LocErrorIro1ComponentOriginServid {
      get {
        return(_LocErrorIro1ComponentOriginServid);
      }
      set {
        _LocErrorIro1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentReturnCode
    /// </summary>
    private char _LocErrorIro1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentReturnCode
    /// </summary>
    public char LocErrorIro1ComponentReturnCode_AS {
      get {
        return(_LocErrorIro1ComponentReturnCode_AS);
      }
      set {
        _LocErrorIro1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocErrorIro1ComponentReturnCode;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentReturnCode
    /// </summary>
    public int LocErrorIro1ComponentReturnCode {
      get {
        return(_LocErrorIro1ComponentReturnCode);
      }
      set {
        _LocErrorIro1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentReasonCode
    /// </summary>
    private char _LocErrorIro1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentReasonCode
    /// </summary>
    public char LocErrorIro1ComponentReasonCode_AS {
      get {
        return(_LocErrorIro1ComponentReasonCode_AS);
      }
      set {
        _LocErrorIro1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocErrorIro1ComponentReasonCode;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentReasonCode
    /// </summary>
    public int LocErrorIro1ComponentReasonCode {
      get {
        return(_LocErrorIro1ComponentReasonCode);
      }
      set {
        _LocErrorIro1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentContextStringTx
    /// </summary>
    private char _LocErrorIro1ComponentContextStringTx_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentContextStringTx
    /// </summary>
    public char LocErrorIro1ComponentContextStringTx_AS {
      get {
        return(_LocErrorIro1ComponentContextStringTx_AS);
      }
      set {
        _LocErrorIro1ComponentContextStringTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentContextStringTx
    /// Domain: Text
    /// Length: 255
    /// Varying Length: Y
    /// </summary>
    private string _LocErrorIro1ComponentContextStringTx;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentContextStringTx
    /// </summary>
    public string LocErrorIro1ComponentContextStringTx {
      get {
        return(_LocErrorIro1ComponentContextStringTx);
      }
      set {
        _LocErrorIro1ComponentContextStringTx = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentSeverityCd
    /// </summary>
    private char _LocErrorIro1ComponentSeverityCd_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentSeverityCd
    /// </summary>
    public char LocErrorIro1ComponentSeverityCd_AS {
      get {
        return(_LocErrorIro1ComponentSeverityCd_AS);
      }
      set {
        _LocErrorIro1ComponentSeverityCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentSeverityCd
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _LocErrorIro1ComponentSeverityCd;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentSeverityCd
    /// </summary>
    public string LocErrorIro1ComponentSeverityCd {
      get {
        return(_LocErrorIro1ComponentSeverityCd);
      }
      set {
        _LocErrorIro1ComponentSeverityCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIro1ComponentRollbackIndicatorTx
    /// </summary>
    private char _LocErrorIro1ComponentRollbackIndicatorTx_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIro1ComponentRollbackIndicatorTx
    /// </summary>
    public char LocErrorIro1ComponentRollbackIndicatorTx_AS {
      get {
        return(_LocErrorIro1ComponentRollbackIndicatorTx_AS);
      }
      set {
        _LocErrorIro1ComponentRollbackIndicatorTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIro1ComponentRollbackIndicatorTx
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _LocErrorIro1ComponentRollbackIndicatorTx;
    /// <summary>
    /// Attribute for: LocErrorIro1ComponentRollbackIndicatorTx
    /// </summary>
    public string LocErrorIro1ComponentRollbackIndicatorTx {
      get {
        return(_LocErrorIro1ComponentRollbackIndicatorTx);
      }
      set {
        _LocErrorIro1ComponentRollbackIndicatorTx = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY9481_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY9481_LA( CYYY9481_LA orig )
    {
      LocImpErrorIro1ComponentSpecificationId_AS = orig.LocImpErrorIro1ComponentSpecificationId_AS;
      LocImpErrorIro1ComponentSpecificationId = orig.LocImpErrorIro1ComponentSpecificationId;
      LocImpErrorIro1ComponentImplementationId_AS = orig.LocImpErrorIro1ComponentImplementationId_AS;
      LocImpErrorIro1ComponentImplementationId = orig.LocImpErrorIro1ComponentImplementationId;
      LocImpErrorIro1ComponentOriginServid_AS = orig.LocImpErrorIro1ComponentOriginServid_AS;
      LocImpErrorIro1ComponentOriginServid = orig.LocImpErrorIro1ComponentOriginServid;
      LocImpErrorIro1ComponentReturnCode_AS = orig.LocImpErrorIro1ComponentReturnCode_AS;
      LocImpErrorIro1ComponentReturnCode = orig.LocImpErrorIro1ComponentReturnCode;
      LocImpErrorIro1ComponentReasonCode_AS = orig.LocImpErrorIro1ComponentReasonCode_AS;
      LocImpErrorIro1ComponentReasonCode = orig.LocImpErrorIro1ComponentReasonCode;
      LocImpErrorIro1ComponentContextStringTx_AS = orig.LocImpErrorIro1ComponentContextStringTx_AS;
      LocImpErrorIro1ComponentContextStringTx = orig.LocImpErrorIro1ComponentContextStringTx;
      LocImpErrorIro1ComponentActivityCd_AS = orig.LocImpErrorIro1ComponentActivityCd_AS;
      LocImpErrorIro1ComponentActivityCd = orig.LocImpErrorIro1ComponentActivityCd;
      LocImpErrorIro1ComponentDialectCd_AS = orig.LocImpErrorIro1ComponentDialectCd_AS;
      LocImpErrorIro1ComponentDialectCd = orig.LocImpErrorIro1ComponentDialectCd;
      LocExpHiliteGroup_MA = orig.LocExpHiliteGroup_MA;
      Array.Copy( orig._LocExpHiliteGroup_AC,
      	LocExpHiliteGroup_AC,
      	LocExpHiliteGroup_AC.Length );
      Array.Copy( orig._LocExpHiliteGIro1ComponentAttributeNameTx_AS,
      	LocExpHiliteGIro1ComponentAttributeNameTx_AS,
      	LocExpHiliteGIro1ComponentAttributeNameTx_AS.Length );
      Array.Copy( orig._LocExpHiliteGIro1ComponentAttributeNameTx,
      	LocExpHiliteGIro1ComponentAttributeNameTx,
      	LocExpHiliteGIro1ComponentAttributeNameTx.Length );
      LocOtherErrorIyy1ComponentReturnCode_AS = orig.LocOtherErrorIyy1ComponentReturnCode_AS;
      LocOtherErrorIyy1ComponentReturnCode = orig.LocOtherErrorIyy1ComponentReturnCode;
      LocOtherErrorIyy1ComponentReasonCode_AS = orig.LocOtherErrorIyy1ComponentReasonCode_AS;
      LocOtherErrorIyy1ComponentReasonCode = orig.LocOtherErrorIyy1ComponentReasonCode;
      LocErrorMsgIro1ComponentSeverityCd_AS = orig.LocErrorMsgIro1ComponentSeverityCd_AS;
      LocErrorMsgIro1ComponentSeverityCd = orig.LocErrorMsgIro1ComponentSeverityCd;
      LocErrorMsgIro1ComponentMessageTx_AS = orig.LocErrorMsgIro1ComponentMessageTx_AS;
      LocErrorMsgIro1ComponentMessageTx = orig.LocErrorMsgIro1ComponentMessageTx;
      LocErrorIro1ComponentOriginServid_AS = orig.LocErrorIro1ComponentOriginServid_AS;
      LocErrorIro1ComponentOriginServid = orig.LocErrorIro1ComponentOriginServid;
      LocErrorIro1ComponentReturnCode_AS = orig.LocErrorIro1ComponentReturnCode_AS;
      LocErrorIro1ComponentReturnCode = orig.LocErrorIro1ComponentReturnCode;
      LocErrorIro1ComponentReasonCode_AS = orig.LocErrorIro1ComponentReasonCode_AS;
      LocErrorIro1ComponentReasonCode = orig.LocErrorIro1ComponentReasonCode;
      LocErrorIro1ComponentContextStringTx_AS = orig.LocErrorIro1ComponentContextStringTx_AS;
      LocErrorIro1ComponentContextStringTx = orig.LocErrorIro1ComponentContextStringTx;
      LocErrorIro1ComponentSeverityCd_AS = orig.LocErrorIro1ComponentSeverityCd_AS;
      LocErrorIro1ComponentSeverityCd = orig.LocErrorIro1ComponentSeverityCd;
      LocErrorIro1ComponentRollbackIndicatorTx_AS = orig.LocErrorIro1ComponentRollbackIndicatorTx_AS;
      LocErrorIro1ComponentRollbackIndicatorTx = orig.LocErrorIro1ComponentRollbackIndicatorTx;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY9481_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY9481_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY9481_LA());
          }
          else 
          {
            CYYY9481_LA result = freeArray[--countFree];
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
      return(new CYYY9481_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocImpErrorIro1ComponentSpecificationId_AS = ' ';
      LocImpErrorIro1ComponentSpecificationId = 0.0;
      LocImpErrorIro1ComponentImplementationId_AS = ' ';
      LocImpErrorIro1ComponentImplementationId = 0.0;
      LocImpErrorIro1ComponentOriginServid_AS = ' ';
      LocImpErrorIro1ComponentOriginServid = 0.0;
      LocImpErrorIro1ComponentReturnCode_AS = ' ';
      LocImpErrorIro1ComponentReturnCode = 0;
      LocImpErrorIro1ComponentReasonCode_AS = ' ';
      LocImpErrorIro1ComponentReasonCode = 0;
      LocImpErrorIro1ComponentContextStringTx_AS = ' ';
      LocImpErrorIro1ComponentContextStringTx = "";
      LocImpErrorIro1ComponentActivityCd_AS = ' ';
      LocImpErrorIro1ComponentActivityCd = "               ";
      LocImpErrorIro1ComponentDialectCd_AS = ' ';
      LocImpErrorIro1ComponentDialectCd = "        ";
      LocExpHiliteGroup_MA = 0;
      for(int a = 0; a < 10; a++)
      {
        LocExpHiliteGroup_AC[ a] = ' ';
        LocExpHiliteGIro1ComponentAttributeNameTx_AS[ a] = ' ';
        LocExpHiliteGIro1ComponentAttributeNameTx[ a] = "                                                  ";
      }
      LocOtherErrorIyy1ComponentReturnCode_AS = ' ';
      LocOtherErrorIyy1ComponentReturnCode = 0;
      LocOtherErrorIyy1ComponentReasonCode_AS = ' ';
      LocOtherErrorIyy1ComponentReasonCode = 0;
      LocErrorMsgIro1ComponentSeverityCd_AS = ' ';
      LocErrorMsgIro1ComponentSeverityCd = " ";
      LocErrorMsgIro1ComponentMessageTx_AS = ' ';
      LocErrorMsgIro1ComponentMessageTx = "";
      LocErrorIro1ComponentOriginServid_AS = ' ';
      LocErrorIro1ComponentOriginServid = 0.0;
      LocErrorIro1ComponentReturnCode_AS = ' ';
      LocErrorIro1ComponentReturnCode = 0;
      LocErrorIro1ComponentReasonCode_AS = ' ';
      LocErrorIro1ComponentReasonCode = 0;
      LocErrorIro1ComponentContextStringTx_AS = ' ';
      LocErrorIro1ComponentContextStringTx = "";
      LocErrorIro1ComponentSeverityCd_AS = ' ';
      LocErrorIro1ComponentSeverityCd = " ";
      LocErrorIro1ComponentRollbackIndicatorTx_AS = ' ';
      LocErrorIro1ComponentRollbackIndicatorTx = " ";
    }
  }
}