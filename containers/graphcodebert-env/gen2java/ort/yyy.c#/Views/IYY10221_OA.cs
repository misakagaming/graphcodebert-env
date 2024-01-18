// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IYY10221_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:58
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
  /// Internal data view storage for: IYY10221_OA
  /// </summary>
  [Serializable]
  public class IYY10221_OA : ViewBase, IExportView
  {
    private static IYY10221_OA[] freeArray = new IYY10221_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCinstanceId
    /// </summary>
    private char _ExpIyy1ChildCinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCinstanceId
    /// </summary>
    public char ExpIyy1ChildCinstanceId_AS {
      get {
        return(_ExpIyy1ChildCinstanceId_AS);
      }
      set {
        _ExpIyy1ChildCinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpIyy1ChildCinstanceId;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCinstanceId
    /// </summary>
    public string ExpIyy1ChildCinstanceId {
      get {
        return(_ExpIyy1ChildCinstanceId);
      }
      set {
        _ExpIyy1ChildCinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCreferenceId
    /// </summary>
    private char _ExpIyy1ChildCreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCreferenceId
    /// </summary>
    public char ExpIyy1ChildCreferenceId_AS {
      get {
        return(_ExpIyy1ChildCreferenceId_AS);
      }
      set {
        _ExpIyy1ChildCreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpIyy1ChildCreferenceId;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCreferenceId
    /// </summary>
    public string ExpIyy1ChildCreferenceId {
      get {
        return(_ExpIyy1ChildCreferenceId);
      }
      set {
        _ExpIyy1ChildCreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCcreateUserId
    /// </summary>
    private char _ExpIyy1ChildCcreateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCcreateUserId
    /// </summary>
    public char ExpIyy1ChildCcreateUserId_AS {
      get {
        return(_ExpIyy1ChildCcreateUserId_AS);
      }
      set {
        _ExpIyy1ChildCcreateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCcreateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1ChildCcreateUserId;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCcreateUserId
    /// </summary>
    public string ExpIyy1ChildCcreateUserId {
      get {
        return(_ExpIyy1ChildCcreateUserId);
      }
      set {
        _ExpIyy1ChildCcreateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCupdateUserId
    /// </summary>
    private char _ExpIyy1ChildCupdateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCupdateUserId
    /// </summary>
    public char ExpIyy1ChildCupdateUserId_AS {
      get {
        return(_ExpIyy1ChildCupdateUserId_AS);
      }
      set {
        _ExpIyy1ChildCupdateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCupdateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1ChildCupdateUserId;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCupdateUserId
    /// </summary>
    public string ExpIyy1ChildCupdateUserId {
      get {
        return(_ExpIyy1ChildCupdateUserId);
      }
      set {
        _ExpIyy1ChildCupdateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCparentPkeyAttrText
    /// </summary>
    private char _ExpIyy1ChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCparentPkeyAttrText
    /// </summary>
    public char ExpIyy1ChildCparentPkeyAttrText_AS {
      get {
        return(_ExpIyy1ChildCparentPkeyAttrText_AS);
      }
      set {
        _ExpIyy1ChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1ChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCparentPkeyAttrText
    /// </summary>
    public string ExpIyy1ChildCparentPkeyAttrText {
      get {
        return(_ExpIyy1ChildCparentPkeyAttrText);
      }
      set {
        _ExpIyy1ChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCkeyAttrNum
    /// </summary>
    private char _ExpIyy1ChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCkeyAttrNum
    /// </summary>
    public char ExpIyy1ChildCkeyAttrNum_AS {
      get {
        return(_ExpIyy1ChildCkeyAttrNum_AS);
      }
      set {
        _ExpIyy1ChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpIyy1ChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCkeyAttrNum
    /// </summary>
    public int ExpIyy1ChildCkeyAttrNum {
      get {
        return(_ExpIyy1ChildCkeyAttrNum);
      }
      set {
        _ExpIyy1ChildCkeyAttrNum = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCsearchAttrText
    /// </summary>
    private char _ExpIyy1ChildCsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCsearchAttrText
    /// </summary>
    public char ExpIyy1ChildCsearchAttrText_AS {
      get {
        return(_ExpIyy1ChildCsearchAttrText_AS);
      }
      set {
        _ExpIyy1ChildCsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1ChildCsearchAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCsearchAttrText
    /// </summary>
    public string ExpIyy1ChildCsearchAttrText {
      get {
        return(_ExpIyy1ChildCsearchAttrText);
      }
      set {
        _ExpIyy1ChildCsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpIyy1ChildCotherAttrText
    /// </summary>
    private char _ExpIyy1ChildCotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ExpIyy1ChildCotherAttrText
    /// </summary>
    public char ExpIyy1ChildCotherAttrText_AS {
      get {
        return(_ExpIyy1ChildCotherAttrText_AS);
      }
      set {
        _ExpIyy1ChildCotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpIyy1ChildCotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ExpIyy1ChildCotherAttrText;
    /// <summary>
    /// Attribute for: ExpIyy1ChildCotherAttrText
    /// </summary>
    public string ExpIyy1ChildCotherAttrText {
      get {
        return(_ExpIyy1ChildCotherAttrText);
      }
      set {
        _ExpIyy1ChildCotherAttrText = value;
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
    
    public IYY10221_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IYY10221_OA( IYY10221_OA orig )
    {
      ExpIyy1ChildCinstanceId_AS = orig.ExpIyy1ChildCinstanceId_AS;
      ExpIyy1ChildCinstanceId = orig.ExpIyy1ChildCinstanceId;
      ExpIyy1ChildCreferenceId_AS = orig.ExpIyy1ChildCreferenceId_AS;
      ExpIyy1ChildCreferenceId = orig.ExpIyy1ChildCreferenceId;
      ExpIyy1ChildCcreateUserId_AS = orig.ExpIyy1ChildCcreateUserId_AS;
      ExpIyy1ChildCcreateUserId = orig.ExpIyy1ChildCcreateUserId;
      ExpIyy1ChildCupdateUserId_AS = orig.ExpIyy1ChildCupdateUserId_AS;
      ExpIyy1ChildCupdateUserId = orig.ExpIyy1ChildCupdateUserId;
      ExpIyy1ChildCparentPkeyAttrText_AS = orig.ExpIyy1ChildCparentPkeyAttrText_AS;
      ExpIyy1ChildCparentPkeyAttrText = orig.ExpIyy1ChildCparentPkeyAttrText;
      ExpIyy1ChildCkeyAttrNum_AS = orig.ExpIyy1ChildCkeyAttrNum_AS;
      ExpIyy1ChildCkeyAttrNum = orig.ExpIyy1ChildCkeyAttrNum;
      ExpIyy1ChildCsearchAttrText_AS = orig.ExpIyy1ChildCsearchAttrText_AS;
      ExpIyy1ChildCsearchAttrText = orig.ExpIyy1ChildCsearchAttrText;
      ExpIyy1ChildCotherAttrText_AS = orig.ExpIyy1ChildCotherAttrText_AS;
      ExpIyy1ChildCotherAttrText = orig.ExpIyy1ChildCotherAttrText;
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
    
    public static IYY10221_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IYY10221_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IYY10221_OA());
          }
          else 
          {
            IYY10221_OA result = freeArray[--countFree];
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
      return(new IYY10221_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpIyy1ChildCinstanceId_AS = ' ';
      ExpIyy1ChildCinstanceId = "00000000000000000000";
      ExpIyy1ChildCreferenceId_AS = ' ';
      ExpIyy1ChildCreferenceId = "00000000000000000000";
      ExpIyy1ChildCcreateUserId_AS = ' ';
      ExpIyy1ChildCcreateUserId = "        ";
      ExpIyy1ChildCupdateUserId_AS = ' ';
      ExpIyy1ChildCupdateUserId = "        ";
      ExpIyy1ChildCparentPkeyAttrText_AS = ' ';
      ExpIyy1ChildCparentPkeyAttrText = "     ";
      ExpIyy1ChildCkeyAttrNum_AS = ' ';
      ExpIyy1ChildCkeyAttrNum = 0;
      ExpIyy1ChildCsearchAttrText_AS = ' ';
      ExpIyy1ChildCsearchAttrText = "                         ";
      ExpIyy1ChildCotherAttrText_AS = ' ';
      ExpIyy1ChildCotherAttrText = "                         ";
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
      this.CopyFrom((IYY10221_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IYY10221_OA orig )
    {
      ExpIyy1ChildCinstanceId_AS = orig.ExpIyy1ChildCinstanceId_AS;
      ExpIyy1ChildCinstanceId = orig.ExpIyy1ChildCinstanceId;
      ExpIyy1ChildCreferenceId_AS = orig.ExpIyy1ChildCreferenceId_AS;
      ExpIyy1ChildCreferenceId = orig.ExpIyy1ChildCreferenceId;
      ExpIyy1ChildCcreateUserId_AS = orig.ExpIyy1ChildCcreateUserId_AS;
      ExpIyy1ChildCcreateUserId = orig.ExpIyy1ChildCcreateUserId;
      ExpIyy1ChildCupdateUserId_AS = orig.ExpIyy1ChildCupdateUserId_AS;
      ExpIyy1ChildCupdateUserId = orig.ExpIyy1ChildCupdateUserId;
      ExpIyy1ChildCparentPkeyAttrText_AS = orig.ExpIyy1ChildCparentPkeyAttrText_AS;
      ExpIyy1ChildCparentPkeyAttrText = orig.ExpIyy1ChildCparentPkeyAttrText;
      ExpIyy1ChildCkeyAttrNum_AS = orig.ExpIyy1ChildCkeyAttrNum_AS;
      ExpIyy1ChildCkeyAttrNum = orig.ExpIyy1ChildCkeyAttrNum;
      ExpIyy1ChildCsearchAttrText_AS = orig.ExpIyy1ChildCsearchAttrText_AS;
      ExpIyy1ChildCsearchAttrText = orig.ExpIyy1ChildCsearchAttrText;
      ExpIyy1ChildCotherAttrText_AS = orig.ExpIyy1ChildCotherAttrText_AS;
      ExpIyy1ChildCotherAttrText = orig.ExpIyy1ChildCotherAttrText;
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
