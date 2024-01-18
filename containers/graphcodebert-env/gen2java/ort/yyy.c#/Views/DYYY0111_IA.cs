// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: DYYY0111_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:51
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
  /// Internal data view storage for: DYYY0111_IA
  /// </summary>
  [Serializable]
  public class DYYY0111_IA : ViewBase, IImportView
  {
    private static DYYY0111_IA[] freeArray = new DYYY0111_IA[30];
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
    // Entity View: IMP_REFERENCE
    //        Type: IYY1_SERVER_DATA
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    private char _ImpReferenceIyy1ServerDataUserid_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public char ImpReferenceIyy1ServerDataUserid_AS {
      get {
        return(_ImpReferenceIyy1ServerDataUserid_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataUserid
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ImpReferenceIyy1ServerDataUserid;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public string ImpReferenceIyy1ServerDataUserid {
      get {
        return(_ImpReferenceIyy1ServerDataUserid);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    private char _ImpReferenceIyy1ServerDataReferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public char ImpReferenceIyy1ServerDataReferenceId_AS {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpReferenceIyy1ServerDataReferenceId;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public string ImpReferenceIyy1ServerDataReferenceId {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId = value;
      }
    }
    // Entity View: IMP
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPinstanceId
    /// </summary>
    private char _ImpParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPinstanceId
    /// </summary>
    public char ImpParentPinstanceId_AS {
      get {
        return(_ImpParentPinstanceId_AS);
      }
      set {
        _ImpParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpParentPinstanceId;
    /// <summary>
    /// Attribute for: ImpParentPinstanceId
    /// </summary>
    public string ImpParentPinstanceId {
      get {
        return(_ImpParentPinstanceId);
      }
      set {
        _ImpParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPkeyAttrText
    /// </summary>
    private char _ImpParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPkeyAttrText
    /// </summary>
    public char ImpParentPkeyAttrText_AS {
      get {
        return(_ImpParentPkeyAttrText_AS);
      }
      set {
        _ImpParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpParentPkeyAttrText
    /// </summary>
    public string ImpParentPkeyAttrText {
      get {
        return(_ImpParentPkeyAttrText);
      }
      set {
        _ImpParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPsearchAttrText
    /// </summary>
    private char _ImpParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPsearchAttrText
    /// </summary>
    public char ImpParentPsearchAttrText_AS {
      get {
        return(_ImpParentPsearchAttrText_AS);
      }
      set {
        _ImpParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPsearchAttrText;
    /// <summary>
    /// Attribute for: ImpParentPsearchAttrText
    /// </summary>
    public string ImpParentPsearchAttrText {
      get {
        return(_ImpParentPsearchAttrText);
      }
      set {
        _ImpParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPotherAttrText
    /// </summary>
    private char _ImpParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPotherAttrText
    /// </summary>
    public char ImpParentPotherAttrText_AS {
      get {
        return(_ImpParentPotherAttrText_AS);
      }
      set {
        _ImpParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPotherAttrText;
    /// <summary>
    /// Attribute for: ImpParentPotherAttrText
    /// </summary>
    public string ImpParentPotherAttrText {
      get {
        return(_ImpParentPotherAttrText);
      }
      set {
        _ImpParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPtypeTkeyAttrText
    /// </summary>
    private char _ImpParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPtypeTkeyAttrText
    /// </summary>
    public char ImpParentPtypeTkeyAttrText_AS {
      get {
        return(_ImpParentPtypeTkeyAttrText_AS);
      }
      set {
        _ImpParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpParentPtypeTkeyAttrText
    /// </summary>
    public string ImpParentPtypeTkeyAttrText {
      get {
        return(_ImpParentPtypeTkeyAttrText);
      }
      set {
        _ImpParentPtypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public DYYY0111_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public DYYY0111_IA( DYYY0111_IA orig )
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
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpParentPinstanceId_AS = orig.ImpParentPinstanceId_AS;
      ImpParentPinstanceId = orig.ImpParentPinstanceId;
      ImpParentPkeyAttrText_AS = orig.ImpParentPkeyAttrText_AS;
      ImpParentPkeyAttrText = orig.ImpParentPkeyAttrText;
      ImpParentPsearchAttrText_AS = orig.ImpParentPsearchAttrText_AS;
      ImpParentPsearchAttrText = orig.ImpParentPsearchAttrText;
      ImpParentPotherAttrText_AS = orig.ImpParentPotherAttrText_AS;
      ImpParentPotherAttrText = orig.ImpParentPotherAttrText;
      ImpParentPtypeTkeyAttrText_AS = orig.ImpParentPtypeTkeyAttrText_AS;
      ImpParentPtypeTkeyAttrText = orig.ImpParentPtypeTkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static DYYY0111_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new DYYY0111_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new DYYY0111_IA());
          }
          else 
          {
            DYYY0111_IA result = freeArray[--countFree];
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
      return(new DYYY0111_IA(this));
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
      ImpReferenceIyy1ServerDataUserid_AS = ' ';
      ImpReferenceIyy1ServerDataUserid = "        ";
      ImpReferenceIyy1ServerDataReferenceId_AS = ' ';
      ImpReferenceIyy1ServerDataReferenceId = "00000000000000000000";
      ImpParentPinstanceId_AS = ' ';
      ImpParentPinstanceId = "00000000000000000000";
      ImpParentPkeyAttrText_AS = ' ';
      ImpParentPkeyAttrText = "     ";
      ImpParentPsearchAttrText_AS = ' ';
      ImpParentPsearchAttrText = "                         ";
      ImpParentPotherAttrText_AS = ' ';
      ImpParentPotherAttrText = "                         ";
      ImpParentPtypeTkeyAttrText_AS = ' ';
      ImpParentPtypeTkeyAttrText = "    ";
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
      this.CopyFrom((DYYY0111_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( DYYY0111_IA orig )
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
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpParentPinstanceId_AS = orig.ImpParentPinstanceId_AS;
      ImpParentPinstanceId = orig.ImpParentPinstanceId;
      ImpParentPkeyAttrText_AS = orig.ImpParentPkeyAttrText_AS;
      ImpParentPkeyAttrText = orig.ImpParentPkeyAttrText;
      ImpParentPsearchAttrText_AS = orig.ImpParentPsearchAttrText_AS;
      ImpParentPsearchAttrText = orig.ImpParentPsearchAttrText;
      ImpParentPotherAttrText_AS = orig.ImpParentPotherAttrText_AS;
      ImpParentPotherAttrText = orig.ImpParentPotherAttrText;
      ImpParentPtypeTkeyAttrText_AS = orig.ImpParentPtypeTkeyAttrText_AS;
      ImpParentPtypeTkeyAttrText = orig.ImpParentPtypeTkeyAttrText;
    }
  }
}
