// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY0111_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:48
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
  /// Internal data view storage for: CYYY0111_LA
  /// </summary>
  [Serializable]
  public class CYYY0111_LA : ViewBase, ILocalView
  {
    private static CYYY0111_LA[] freeArray = new CYYY0111_LA[30];
    private static int countFree = 0;
    
    // Entity View: LOC_IMP
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPinstanceId
    /// </summary>
    private char _LocImpParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPinstanceId
    /// </summary>
    public char LocImpParentPinstanceId_AS {
      get {
        return(_LocImpParentPinstanceId_AS);
      }
      set {
        _LocImpParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _LocImpParentPinstanceId;
    /// <summary>
    /// Attribute for: LocImpParentPinstanceId
    /// </summary>
    public string LocImpParentPinstanceId {
      get {
        return(_LocImpParentPinstanceId);
      }
      set {
        _LocImpParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPkeyAttrText
    /// </summary>
    private char _LocImpParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPkeyAttrText
    /// </summary>
    public char LocImpParentPkeyAttrText_AS {
      get {
        return(_LocImpParentPkeyAttrText_AS);
      }
      set {
        _LocImpParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _LocImpParentPkeyAttrText;
    /// <summary>
    /// Attribute for: LocImpParentPkeyAttrText
    /// </summary>
    public string LocImpParentPkeyAttrText {
      get {
        return(_LocImpParentPkeyAttrText);
      }
      set {
        _LocImpParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPsearchAttrText
    /// </summary>
    private char _LocImpParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPsearchAttrText
    /// </summary>
    public char LocImpParentPsearchAttrText_AS {
      get {
        return(_LocImpParentPsearchAttrText_AS);
      }
      set {
        _LocImpParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocImpParentPsearchAttrText;
    /// <summary>
    /// Attribute for: LocImpParentPsearchAttrText
    /// </summary>
    public string LocImpParentPsearchAttrText {
      get {
        return(_LocImpParentPsearchAttrText);
      }
      set {
        _LocImpParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPotherAttrText
    /// </summary>
    private char _LocImpParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPotherAttrText
    /// </summary>
    public char LocImpParentPotherAttrText_AS {
      get {
        return(_LocImpParentPotherAttrText_AS);
      }
      set {
        _LocImpParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocImpParentPotherAttrText;
    /// <summary>
    /// Attribute for: LocImpParentPotherAttrText
    /// </summary>
    public string LocImpParentPotherAttrText {
      get {
        return(_LocImpParentPotherAttrText);
      }
      set {
        _LocImpParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPtypeTkeyAttrText
    /// </summary>
    private char _LocImpParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPtypeTkeyAttrText
    /// </summary>
    public char LocImpParentPtypeTkeyAttrText_AS {
      get {
        return(_LocImpParentPtypeTkeyAttrText_AS);
      }
      set {
        _LocImpParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _LocImpParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: LocImpParentPtypeTkeyAttrText
    /// </summary>
    public string LocImpParentPtypeTkeyAttrText {
      get {
        return(_LocImpParentPtypeTkeyAttrText);
      }
      set {
        _LocImpParentPtypeTkeyAttrText = value;
      }
    }
    // Entity View: LOC_EMPTY
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocEmptyParentPinstanceId
    /// </summary>
    private char _LocEmptyParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: LocEmptyParentPinstanceId
    /// </summary>
    public char LocEmptyParentPinstanceId_AS {
      get {
        return(_LocEmptyParentPinstanceId_AS);
      }
      set {
        _LocEmptyParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocEmptyParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _LocEmptyParentPinstanceId;
    /// <summary>
    /// Attribute for: LocEmptyParentPinstanceId
    /// </summary>
    public string LocEmptyParentPinstanceId {
      get {
        return(_LocEmptyParentPinstanceId);
      }
      set {
        _LocEmptyParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocEmptyParentPkeyAttrText
    /// </summary>
    private char _LocEmptyParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocEmptyParentPkeyAttrText
    /// </summary>
    public char LocEmptyParentPkeyAttrText_AS {
      get {
        return(_LocEmptyParentPkeyAttrText_AS);
      }
      set {
        _LocEmptyParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocEmptyParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _LocEmptyParentPkeyAttrText;
    /// <summary>
    /// Attribute for: LocEmptyParentPkeyAttrText
    /// </summary>
    public string LocEmptyParentPkeyAttrText {
      get {
        return(_LocEmptyParentPkeyAttrText);
      }
      set {
        _LocEmptyParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocEmptyParentPsearchAttrText
    /// </summary>
    private char _LocEmptyParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocEmptyParentPsearchAttrText
    /// </summary>
    public char LocEmptyParentPsearchAttrText_AS {
      get {
        return(_LocEmptyParentPsearchAttrText_AS);
      }
      set {
        _LocEmptyParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocEmptyParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocEmptyParentPsearchAttrText;
    /// <summary>
    /// Attribute for: LocEmptyParentPsearchAttrText
    /// </summary>
    public string LocEmptyParentPsearchAttrText {
      get {
        return(_LocEmptyParentPsearchAttrText);
      }
      set {
        _LocEmptyParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocEmptyParentPotherAttrText
    /// </summary>
    private char _LocEmptyParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocEmptyParentPotherAttrText
    /// </summary>
    public char LocEmptyParentPotherAttrText_AS {
      get {
        return(_LocEmptyParentPotherAttrText_AS);
      }
      set {
        _LocEmptyParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocEmptyParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocEmptyParentPotherAttrText;
    /// <summary>
    /// Attribute for: LocEmptyParentPotherAttrText
    /// </summary>
    public string LocEmptyParentPotherAttrText {
      get {
        return(_LocEmptyParentPotherAttrText);
      }
      set {
        _LocEmptyParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocEmptyParentPtypeTkeyAttrText
    /// </summary>
    private char _LocEmptyParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocEmptyParentPtypeTkeyAttrText
    /// </summary>
    public char LocEmptyParentPtypeTkeyAttrText_AS {
      get {
        return(_LocEmptyParentPtypeTkeyAttrText_AS);
      }
      set {
        _LocEmptyParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocEmptyParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _LocEmptyParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: LocEmptyParentPtypeTkeyAttrText
    /// </summary>
    public string LocEmptyParentPtypeTkeyAttrText {
      get {
        return(_LocEmptyParentPtypeTkeyAttrText);
      }
      set {
        _LocEmptyParentPtypeTkeyAttrText = value;
      }
    }
    // Entity View: LOC_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentSeverityCode
    /// </summary>
    private char _LocErrorIyy1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentSeverityCode
    /// </summary>
    public char LocErrorIyy1ComponentSeverityCode_AS {
      get {
        return(_LocErrorIyy1ComponentSeverityCode_AS);
      }
      set {
        _LocErrorIyy1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _LocErrorIyy1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentSeverityCode
    /// </summary>
    public string LocErrorIyy1ComponentSeverityCode {
      get {
        return(_LocErrorIyy1ComponentSeverityCode);
      }
      set {
        _LocErrorIyy1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentRollbackIndicator
    /// </summary>
    private char _LocErrorIyy1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public char LocErrorIyy1ComponentRollbackIndicator_AS {
      get {
        return(_LocErrorIyy1ComponentRollbackIndicator_AS);
      }
      set {
        _LocErrorIyy1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _LocErrorIyy1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public string LocErrorIyy1ComponentRollbackIndicator {
      get {
        return(_LocErrorIyy1ComponentRollbackIndicator);
      }
      set {
        _LocErrorIyy1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentOriginServid
    /// </summary>
    private char _LocErrorIyy1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentOriginServid
    /// </summary>
    public char LocErrorIyy1ComponentOriginServid_AS {
      get {
        return(_LocErrorIyy1ComponentOriginServid_AS);
      }
      set {
        _LocErrorIyy1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _LocErrorIyy1ComponentOriginServid;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentOriginServid
    /// </summary>
    public double LocErrorIyy1ComponentOriginServid {
      get {
        return(_LocErrorIyy1ComponentOriginServid);
      }
      set {
        _LocErrorIyy1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentContextString
    /// </summary>
    private char _LocErrorIyy1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentContextString
    /// </summary>
    public char LocErrorIyy1ComponentContextString_AS {
      get {
        return(_LocErrorIyy1ComponentContextString_AS);
      }
      set {
        _LocErrorIyy1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _LocErrorIyy1ComponentContextString;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentContextString
    /// </summary>
    public string LocErrorIyy1ComponentContextString {
      get {
        return(_LocErrorIyy1ComponentContextString);
      }
      set {
        _LocErrorIyy1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentReturnCode
    /// </summary>
    private char _LocErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentReturnCode
    /// </summary>
    public char LocErrorIyy1ComponentReturnCode_AS {
      get {
        return(_LocErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _LocErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentReturnCode
    /// </summary>
    public int LocErrorIyy1ComponentReturnCode {
      get {
        return(_LocErrorIyy1ComponentReturnCode);
      }
      set {
        _LocErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentReasonCode
    /// </summary>
    private char _LocErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentReasonCode
    /// </summary>
    public char LocErrorIyy1ComponentReasonCode_AS {
      get {
        return(_LocErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _LocErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentReasonCode
    /// </summary>
    public int LocErrorIyy1ComponentReasonCode {
      get {
        return(_LocErrorIyy1ComponentReasonCode);
      }
      set {
        _LocErrorIyy1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocErrorIyy1ComponentChecksum
    /// </summary>
    private char _LocErrorIyy1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: LocErrorIyy1ComponentChecksum
    /// </summary>
    public char LocErrorIyy1ComponentChecksum_AS {
      get {
        return(_LocErrorIyy1ComponentChecksum_AS);
      }
      set {
        _LocErrorIyy1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocErrorIyy1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _LocErrorIyy1ComponentChecksum;
    /// <summary>
    /// Attribute for: LocErrorIyy1ComponentChecksum
    /// </summary>
    public string LocErrorIyy1ComponentChecksum {
      get {
        return(_LocErrorIyy1ComponentChecksum);
      }
      set {
        _LocErrorIyy1ComponentChecksum = value;
      }
    }
    // Repeating GV:  LOC_GROUP_CONTEXT
    //     Repeats: 9 times
    /// <summary>
    /// Internal storage, repeating group view count
    /// </summary>
    private int _LocGroupContext_MA;
    /// <summary>
    /// Repeating group view count
    /// </summary>
    public int LocGroupContext_MA {
      get {
        return(_LocGroupContext_MA);
      }
      set {
        _LocGroupContext_MA = value;
      }
    }
    /// <summary>
    /// Internal storage, repeating group view occurrance array
    /// </summary>
    private char[] _LocGroupContext_AC = new char[9];
    /// <summary>
    /// Repeating group view occurrance array
    /// </summary>
    public char[] LocGroupContext_AC {
      get {
        return(_LocGroupContext_AC);
      }
      set {
        _LocGroupContext_AC = value;
      }
    }
    // Entity View: LOC_G_CONTEXT
    //        Type: DONT_CHANGE_TEXT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocGContextDontChangeTextText150
    /// </summary>
    private char[] _LocGContextDontChangeTextText150_AS = new char[9];
    /// <summary>
    /// Attribute missing flag for: LocGContextDontChangeTextText150
    /// </summary>
    public char[] LocGContextDontChangeTextText150_AS {
      get {
        return(_LocGContextDontChangeTextText150_AS);
      }
      set {
        _LocGContextDontChangeTextText150_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocGContextDontChangeTextText150
    /// Domain: Text
    /// Length: 150
    /// Varying Length: N
    /// </summary>
    private string[] _LocGContextDontChangeTextText150 = new string[9];
    /// <summary>
    /// Attribute for: LocGContextDontChangeTextText150
    /// </summary>
    public string[] LocGContextDontChangeTextText150 {
      get {
        return(_LocGContextDontChangeTextText150);
      }
      set {
        _LocGContextDontChangeTextText150 = value;
      }
    }
    // Entity View: LOC
    //        Type: DONT_CHANGE_RETURN_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    private char _LocDontChangeReturnCodesQ1Ok_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public char LocDontChangeReturnCodesQ1Ok_AS {
      get {
        return(_LocDontChangeReturnCodesQ1Ok_AS);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesQ1Ok
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesQ1Ok;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public int LocDontChangeReturnCodesQ1Ok {
      get {
        return(_LocDontChangeReturnCodesQ1Ok);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN20MandatoryImportMissing
    /// </summary>
    private char _LocDontChangeReturnCodesN20MandatoryImportMissing_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN20MandatoryImportMissing
    /// </summary>
    public char LocDontChangeReturnCodesN20MandatoryImportMissing_AS {
      get {
        return(_LocDontChangeReturnCodesN20MandatoryImportMissing_AS);
      }
      set {
        _LocDontChangeReturnCodesN20MandatoryImportMissing_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN20MandatoryImportMissing
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN20MandatoryImportMissing;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN20MandatoryImportMissing
    /// </summary>
    public int LocDontChangeReturnCodesN20MandatoryImportMissing {
      get {
        return(_LocDontChangeReturnCodesN20MandatoryImportMissing);
      }
      set {
        _LocDontChangeReturnCodesN20MandatoryImportMissing = value;
      }
    }
    // Entity View: LOC
    //        Type: DONT_CHANGE_REASON_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    private char _LocDontChangeReasonCodesQ1Default_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public char LocDontChangeReasonCodesQ1Default_AS {
      get {
        return(_LocDontChangeReasonCodesQ1Default_AS);
      }
      set {
        _LocDontChangeReasonCodesQ1Default_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ1Default
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ1Default;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public int LocDontChangeReasonCodesQ1Default {
      get {
        return(_LocDontChangeReasonCodesQ1Default);
      }
      set {
        _LocDontChangeReasonCodesQ1Default = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand
    /// </summary>
    private char _LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand
    /// </summary>
    public char LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS {
      get {
        return(_LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS);
      }
      set {
        _LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand
    /// </summary>
    public int LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand {
      get {
        return(_LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand);
      }
      set {
        _LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ105ParentKeyAttrMand
    /// </summary>
    private char _LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ105ParentKeyAttrMand
    /// </summary>
    public char LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS {
      get {
        return(_LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS);
      }
      set {
        _LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ105ParentKeyAttrMand
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ105ParentKeyAttrMand;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ105ParentKeyAttrMand
    /// </summary>
    public int LocDontChangeReasonCodesQ105ParentKeyAttrMand {
      get {
        return(_LocDontChangeReasonCodesQ105ParentKeyAttrMand);
      }
      set {
        _LocDontChangeReasonCodesQ105ParentKeyAttrMand = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ106ParentSearchAttrMand
    /// </summary>
    private char _LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ106ParentSearchAttrMand
    /// </summary>
    public char LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS {
      get {
        return(_LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS);
      }
      set {
        _LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ106ParentSearchAttrMand
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ106ParentSearchAttrMand;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ106ParentSearchAttrMand
    /// </summary>
    public int LocDontChangeReasonCodesQ106ParentSearchAttrMand {
      get {
        return(_LocDontChangeReasonCodesQ106ParentSearchAttrMand);
      }
      set {
        _LocDontChangeReasonCodesQ106ParentSearchAttrMand = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY0111_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY0111_LA( CYYY0111_LA orig )
    {
      LocImpParentPinstanceId_AS = orig.LocImpParentPinstanceId_AS;
      LocImpParentPinstanceId = orig.LocImpParentPinstanceId;
      LocImpParentPkeyAttrText_AS = orig.LocImpParentPkeyAttrText_AS;
      LocImpParentPkeyAttrText = orig.LocImpParentPkeyAttrText;
      LocImpParentPsearchAttrText_AS = orig.LocImpParentPsearchAttrText_AS;
      LocImpParentPsearchAttrText = orig.LocImpParentPsearchAttrText;
      LocImpParentPotherAttrText_AS = orig.LocImpParentPotherAttrText_AS;
      LocImpParentPotherAttrText = orig.LocImpParentPotherAttrText;
      LocImpParentPtypeTkeyAttrText_AS = orig.LocImpParentPtypeTkeyAttrText_AS;
      LocImpParentPtypeTkeyAttrText = orig.LocImpParentPtypeTkeyAttrText;
      LocEmptyParentPinstanceId_AS = orig.LocEmptyParentPinstanceId_AS;
      LocEmptyParentPinstanceId = orig.LocEmptyParentPinstanceId;
      LocEmptyParentPkeyAttrText_AS = orig.LocEmptyParentPkeyAttrText_AS;
      LocEmptyParentPkeyAttrText = orig.LocEmptyParentPkeyAttrText;
      LocEmptyParentPsearchAttrText_AS = orig.LocEmptyParentPsearchAttrText_AS;
      LocEmptyParentPsearchAttrText = orig.LocEmptyParentPsearchAttrText;
      LocEmptyParentPotherAttrText_AS = orig.LocEmptyParentPotherAttrText_AS;
      LocEmptyParentPotherAttrText = orig.LocEmptyParentPotherAttrText;
      LocEmptyParentPtypeTkeyAttrText_AS = orig.LocEmptyParentPtypeTkeyAttrText_AS;
      LocEmptyParentPtypeTkeyAttrText = orig.LocEmptyParentPtypeTkeyAttrText;
      LocErrorIyy1ComponentSeverityCode_AS = orig.LocErrorIyy1ComponentSeverityCode_AS;
      LocErrorIyy1ComponentSeverityCode = orig.LocErrorIyy1ComponentSeverityCode;
      LocErrorIyy1ComponentRollbackIndicator_AS = orig.LocErrorIyy1ComponentRollbackIndicator_AS;
      LocErrorIyy1ComponentRollbackIndicator = orig.LocErrorIyy1ComponentRollbackIndicator;
      LocErrorIyy1ComponentOriginServid_AS = orig.LocErrorIyy1ComponentOriginServid_AS;
      LocErrorIyy1ComponentOriginServid = orig.LocErrorIyy1ComponentOriginServid;
      LocErrorIyy1ComponentContextString_AS = orig.LocErrorIyy1ComponentContextString_AS;
      LocErrorIyy1ComponentContextString = orig.LocErrorIyy1ComponentContextString;
      LocErrorIyy1ComponentReturnCode_AS = orig.LocErrorIyy1ComponentReturnCode_AS;
      LocErrorIyy1ComponentReturnCode = orig.LocErrorIyy1ComponentReturnCode;
      LocErrorIyy1ComponentReasonCode_AS = orig.LocErrorIyy1ComponentReasonCode_AS;
      LocErrorIyy1ComponentReasonCode = orig.LocErrorIyy1ComponentReasonCode;
      LocErrorIyy1ComponentChecksum_AS = orig.LocErrorIyy1ComponentChecksum_AS;
      LocErrorIyy1ComponentChecksum = orig.LocErrorIyy1ComponentChecksum;
      LocGroupContext_MA = orig.LocGroupContext_MA;
      Array.Copy( orig._LocGroupContext_AC,
      	LocGroupContext_AC,
      	LocGroupContext_AC.Length );
      Array.Copy( orig._LocGContextDontChangeTextText150_AS,
      	LocGContextDontChangeTextText150_AS,
      	LocGContextDontChangeTextText150_AS.Length );
      Array.Copy( orig._LocGContextDontChangeTextText150,
      	LocGContextDontChangeTextText150,
      	LocGContextDontChangeTextText150.Length );
      LocDontChangeReturnCodesQ1Ok_AS = orig.LocDontChangeReturnCodesQ1Ok_AS;
      LocDontChangeReturnCodesQ1Ok = orig.LocDontChangeReturnCodesQ1Ok;
      LocDontChangeReturnCodesN20MandatoryImportMissing_AS = orig.LocDontChangeReturnCodesN20MandatoryImportMissing_AS;
      LocDontChangeReturnCodesN20MandatoryImportMissing = orig.LocDontChangeReturnCodesN20MandatoryImportMissing;
      LocDontChangeReasonCodesQ1Default_AS = orig.LocDontChangeReasonCodesQ1Default_AS;
      LocDontChangeReasonCodesQ1Default = orig.LocDontChangeReasonCodesQ1Default;
      LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS = orig.LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS;
      LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand = orig.LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand;
      LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS = orig.LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS;
      LocDontChangeReasonCodesQ105ParentKeyAttrMand = orig.LocDontChangeReasonCodesQ105ParentKeyAttrMand;
      LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS = orig.LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS;
      LocDontChangeReasonCodesQ106ParentSearchAttrMand = orig.LocDontChangeReasonCodesQ106ParentSearchAttrMand;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY0111_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY0111_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY0111_LA());
          }
          else 
          {
            CYYY0111_LA result = freeArray[--countFree];
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
      return(new CYYY0111_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocImpParentPinstanceId_AS = ' ';
      LocImpParentPinstanceId = "00000000000000000000";
      LocImpParentPkeyAttrText_AS = ' ';
      LocImpParentPkeyAttrText = "     ";
      LocImpParentPsearchAttrText_AS = ' ';
      LocImpParentPsearchAttrText = "                         ";
      LocImpParentPotherAttrText_AS = ' ';
      LocImpParentPotherAttrText = "                         ";
      LocImpParentPtypeTkeyAttrText_AS = ' ';
      LocImpParentPtypeTkeyAttrText = "    ";
      LocEmptyParentPinstanceId_AS = ' ';
      LocEmptyParentPinstanceId = "00000000000000000000";
      LocEmptyParentPkeyAttrText_AS = ' ';
      LocEmptyParentPkeyAttrText = "     ";
      LocEmptyParentPsearchAttrText_AS = ' ';
      LocEmptyParentPsearchAttrText = "                         ";
      LocEmptyParentPotherAttrText_AS = ' ';
      LocEmptyParentPotherAttrText = "                         ";
      LocEmptyParentPtypeTkeyAttrText_AS = ' ';
      LocEmptyParentPtypeTkeyAttrText = "    ";
      LocErrorIyy1ComponentSeverityCode_AS = ' ';
      LocErrorIyy1ComponentSeverityCode = " ";
      LocErrorIyy1ComponentRollbackIndicator_AS = ' ';
      LocErrorIyy1ComponentRollbackIndicator = " ";
      LocErrorIyy1ComponentOriginServid_AS = ' ';
      LocErrorIyy1ComponentOriginServid = 0.0;
      LocErrorIyy1ComponentContextString_AS = ' ';
      LocErrorIyy1ComponentContextString = "";
      LocErrorIyy1ComponentReturnCode_AS = ' ';
      LocErrorIyy1ComponentReturnCode = 0;
      LocErrorIyy1ComponentReasonCode_AS = ' ';
      LocErrorIyy1ComponentReasonCode = 0;
      LocErrorIyy1ComponentChecksum_AS = ' ';
      LocErrorIyy1ComponentChecksum = "               ";
      LocGroupContext_MA = 0;
      for(int a = 0; a < 9; a++)
      {
        LocGroupContext_AC[ a] = ' ';
        LocGContextDontChangeTextText150_AS[ a] = ' ';
        LocGContextDontChangeTextText150[ a] = 
"                                                                                                                                                      "
          ;
      }
      LocDontChangeReturnCodesQ1Ok_AS = ' ';
      LocDontChangeReturnCodesQ1Ok = 0;
      LocDontChangeReturnCodesN20MandatoryImportMissing_AS = ' ';
      LocDontChangeReturnCodesN20MandatoryImportMissing = 0;
      LocDontChangeReasonCodesQ1Default_AS = ' ';
      LocDontChangeReasonCodesQ1Default = 0;
      LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand_AS = ' ';
      LocDontChangeReasonCodesQ104ParentTypeKeyAttrMand = 0;
      LocDontChangeReasonCodesQ105ParentKeyAttrMand_AS = ' ';
      LocDontChangeReasonCodesQ105ParentKeyAttrMand = 0;
      LocDontChangeReasonCodesQ106ParentSearchAttrMand_AS = ' ';
      LocDontChangeReasonCodesQ106ParentSearchAttrMand = 0;
    }
  }
}
